#!/usr/bin/python

import sys, time, os, pdb, argparse, pickle, subprocess, glob, cv2, tempfile
import numpy as np
from shutil import rmtree

import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy import signal

from detectors import S3FD

# ========== ========== ========== ==========
# # PARSE ARGS
# ========== ========== ========== ==========

parser = argparse.ArgumentParser(description="FaceTracker")
parser.add_argument(
    "--data_dir", type=str, default="data/work", help="Output directory"
)
parser.add_argument("--videofile", type=str, default="", help="Input video file")
parser.add_argument("--reference", type=str, default="", help="Video reference")
parser.add_argument(
    "--facedet_scale", type=float, default=0.25, help="Scale factor for face detection"
)
parser.add_argument("--crop_scale", type=float, default=0.40, help="Scale bounding box")
parser.add_argument(
    "--min_track", type=int, default=100, help="Minimum facetrack duration"
)
parser.add_argument("--frame_rate", type=int, default=25, help="Frame rate")
parser.add_argument(
    "--num_failed_det",
    type=int,
    default=25,
    help="Number of missed detections allowed before tracking is stopped",
)
parser.add_argument(
    "--min_face_size", type=int, default=100, help="Minimum face size in pixels"
)
parser.add_argument(
    "--chunk_size", type=int, default=300, help="Number of seconds per video chunk"
)
opt = parser.parse_args()

setattr(opt, "avi_dir", os.path.join(opt.data_dir, "pyavi"))
setattr(opt, "tmp_dir", os.path.join(opt.data_dir, "pytmp"))
setattr(opt, "work_dir", os.path.join(opt.data_dir, "pywork"))
setattr(opt, "crop_dir", os.path.join(opt.data_dir, "pycrop"))
setattr(opt, "frames_dir", os.path.join(opt.data_dir, "pyframes"))


# ========== ========== ========== ==========
# # IOU FUNCTION
# ========== ========== ========== ==========


def bb_intersection_over_union(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


# ========== ========== ========== ==========
# # VIDEO CHUNKING
# ========== ========== ========== ==========


def split_video_into_chunks(opt):
    # Get video duration using ffprobe
    command = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {opt.videofile}"
    duration = float(subprocess.check_output(command, shell=True).decode().strip())

    chunk_starts = list(range(0, int(duration), opt.chunk_size))
    chunk_files = []

    os.makedirs(os.path.join(opt.avi_dir, opt.reference), exist_ok=True)

    for i, start in enumerate(chunk_starts):
        chunk_file = os.path.join(opt.avi_dir, opt.reference, f"chunk_{i:03d}.avi")
        duration_arg = (
            f"-t {opt.chunk_size}" if start + opt.chunk_size < duration else ""
        )
        command = (
            f"ffmpeg -y -i {opt.videofile} -ss {start} {duration_arg} "
            f"-qscale:v 2 -async 1 -r {opt.frame_rate} {chunk_file}"
        )
        subprocess.call(command, shell=True, stdout=None)
        chunk_files.append(chunk_file)

    return chunk_files


# ========== ========== ========== ==========
# # FACE TRACKING
# ========== ========== ========== ==========


def track_shot(opt, scenefaces):

    iouThres = 0.5  # Minimum IOU between consecutive face detections
    tracks = []

    while True:
        track = []
        for framefaces in scenefaces:
            for face in framefaces:
                if track == []:
                    track.append(face)
                    framefaces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= opt.num_failed_det:
                    iou = bb_intersection_over_union(face["bbox"], track[-1]["bbox"])
                    if iou > iouThres:
                        track.append(face)
                        framefaces.remove(face)
                        continue
                else:
                    break

        if track == []:
            break
        elif len(track) > opt.min_track:

            framenum = np.array([f["frame"] for f in track])
            bboxes = np.array([np.array(f["bbox"]) for f in track])

            frame_i = np.arange(framenum[0], framenum[-1] + 1)

            bboxes_i = []
            for ij in range(0, 4):
                interpfn = interp1d(framenum, bboxes[:, ij])
                bboxes_i.append(interpfn(frame_i))
            bboxes_i = np.stack(bboxes_i, axis=1)

            if (
                max(
                    np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]),
                    np.mean(bboxes_i[:, 3] - bboxes_i[:, 1]),
                )
                > opt.min_face_size
            ):
                tracks.append({"frame": frame_i, "bbox": bboxes_i})

    return tracks


# ========== ========== ========== ==========
# # VIDEO CROP AND SAVE
# ========== ========== ========== ==========


def crop_video(opt, track, cropfile, chunk_idx):

    flist = glob.glob(
        os.path.join(opt.frames_dir, opt.reference, f"chunk_{chunk_idx:03d}", "*.jpg")
    )
    flist.sort()

    # Calculate the maximum face dimensions
    max_face_width = 0
    max_face_height = 0

    for det in track["bbox"]:
        width = int((det[2] - det[0]) * (1 + 2 * opt.crop_scale))
        height = int((det[3] - det[1]) * (1 + 2 * opt.crop_scale))
        max_face_width = max(max_face_width, width)
        max_face_height = max(max_face_height, height)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vOut = cv2.VideoWriter(
        cropfile + "t.avi", fourcc, opt.frame_rate, (max_face_width, max_face_height)
    )

    dets = {"x": [], "y": [], "s": []}

    for det in track["bbox"]:
        dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets["y"].append((det[1] + det[3]) / 2)
        dets["x"].append((det[0] + det[2]) / 2)

    dets["s"] = signal.medfilt(dets["s"], kernel_size=13)
    dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
    dets["y"] = signal.medfilt(dets["y"], kernel_size=13)

    for fidx, frame in enumerate(track["frame"]):
        cs = opt.crop_scale
        bs = dets["s"][fidx]
        bsi = int(bs * (1 + 2 * cs))

        image = cv2.imread(flist[frame])

        frame = np.pad(
            image,
            ((bsi, bsi), (bsi, bsi), (0, 0)),
            "constant",
            constant_values=(110, 110),
        )
        my = dets["y"][fidx] + bsi
        mx = dets["x"][fidx] + bsi

        face = frame[
            int(my - bs) : int(my + bs * (1 + 2 * cs)),
            int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs)),
        ]

        face_h, face_w = face.shape[:2]
        scale = min(max_face_height / face_h, max_face_width / face_w)
        new_h, new_w = int(face_h * scale), int(face_w * scale)

        resized_face = cv2.resize(face, (new_w, new_h))
        final_face = (
            np.zeros((max_face_height, max_face_width, 3), dtype=np.uint8) + 110
        )
        y_offset = (max_face_height - new_h) // 2
        x_offset = (max_face_width - new_w) // 2
        final_face[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = (
            resized_face
        )

        vOut.write(final_face)

    audiotmp = os.path.join(opt.tmp_dir, opt.reference, f"audio_{chunk_idx:03d}.wav")
    audiostart = (track["frame"][0]) / opt.frame_rate
    audioend = (track["frame"][-1] + 1) / opt.frame_rate

    vOut.release()

    command = (
        f"ffmpeg -y -i {os.path.join(opt.avi_dir,opt.reference,f'chunk_{chunk_idx:03d}.avi')} "
        f"-ss {audiostart:.3f} -to {audioend:.3f} {audiotmp}"
    )
    subprocess.call(command, shell=True, stdout=None)

    command = (
        f"ffmpeg -y -i {cropfile}t.avi -i {audiotmp} -c:v copy -c:a copy {cropfile}.avi"
    )
    subprocess.call(command, shell=True, stdout=None)

    os.remove(cropfile + "t.avi")
    print(f"Written {cropfile}")
    print(
        f'Mean pos: x {np.mean(dets["x"]):.2f} y {np.mean(dets["y"]):.2f} s {np.mean(dets["s"]):.2f}'
    )

    return {"track": track, "proc_track": dets}


# ========== ========== ========== ==========
# # FACE DETECTION
# ========== ========== ========== ==========


def inference_video(opt, chunk_file, chunk_idx):
    DET = S3FD(device="cuda")

    flist = glob.glob(
        os.path.join(opt.frames_dir, opt.reference, f"chunk_{chunk_idx:03d}", "*.jpg")
    )
    flist.sort()

    dets = []
    for fidx, fname in enumerate(flist):
        start_time = time.time()
        image = cv2.imread(fname)
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[opt.facedet_scale])

        dets.append([])
        for bbox in bboxes:
            dets[-1].append(
                {"frame": fidx, "bbox": (bbox[:-1]).tolist(), "conf": bbox[-1]}
            )

        elapsed_time = time.time() - start_time
        print(f"{chunk_file}-{fidx:05d}; {len(dets[-1])} dets; {1/elapsed_time:.2f} Hz")

    savepath = os.path.join(opt.work_dir, opt.reference, f"faces_{chunk_idx:03d}.pckl")
    with open(savepath, "wb") as fil:
        pickle.dump(dets, fil)

    return dets


# ========== ========== ========== ==========
# # SCENE DETECTION
# ========== ========== ========== ==========


def scene_detect(opt, chunk_file, chunk_idx):
    video_manager = VideoManager([chunk_file])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()

    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    scene_list = scene_manager.get_scene_list(base_timecode)
    savepath = os.path.join(opt.work_dir, opt.reference, f"scene_{chunk_idx:03d}.pckl")

    if scene_list == []:
        scene_list = [
            (video_manager.get_base_timecode(), video_manager.get_current_timecode())
        ]

    with open(savepath, "wb") as fil:
        pickle.dump(scene_list, fil)

    print(f"{chunk_file} - scenes detected {len(scene_list)}")
    return scene_list


# ========== ========== ========== ==========
# # PROCESS CHUNK
# ========== ========== ========== ==========


def process_chunk(opt, chunk_file, chunk_idx):
    # Create chunk-specific directories
    os.makedirs(os.path.join(opt.work_dir, opt.reference), exist_ok=True)
    os.makedirs(os.path.join(opt.crop_dir, opt.reference), exist_ok=True)
    os.makedirs(
        os.path.join(opt.frames_dir, opt.reference, f"chunk_{chunk_idx:03d}"),
        exist_ok=True,
    )
    os.makedirs(os.path.join(opt.tmp_dir, opt.reference), exist_ok=True)

    # Extract frames
    command = (
        f"ffmpeg -y -i {chunk_file} -qscale:v 2 -threads 1 -f image2 "
        f"{os.path.join(opt.frames_dir,opt.reference,f'chunk_{chunk_idx:03d}','%06d.jpg')}"
    )
    subprocess.call(command, shell=True, stdout=None)

    # Face detection
    faces = inference_video(opt, chunk_file, chunk_idx)

    # Scene detection
    scenes = scene_detect(opt, chunk_file, chunk_idx)

    # Face tracking
    alltracks = []
    vidtracks = []
    for shot in scenes:
        if shot[1].frame_num - shot[0].frame_num >= opt.min_track:
            alltracks.extend(
                track_shot(opt, faces[shot[0].frame_num : shot[1].frame_num])
            )

    # Face track crop
    for ii, track in enumerate(alltracks):
        vidtracks.append(
            crop_video(
                opt,
                track,
                os.path.join(opt.crop_dir, opt.reference, f"{chunk_idx:03d}_{ii:05d}"),
                chunk_idx,
            )
        )

    return vidtracks


def merge_face_tracks(opt, output_file=None):
    # If no output file specified, create one
    if output_file is None:
        output_file = os.path.join(opt.crop_dir, f"{opt.reference}_merged.avi")

    # Get all the cropped face track videos
    crop_files = glob.glob(os.path.join(opt.crop_dir, opt.reference, "*.avi"))

    if not crop_files:
        print(f"No video files found in {os.path.join(opt.crop_dir, opt.reference)}")
        return None

    # Sort the files to maintain order
    crop_files.sort()

    # Create a temporary file listing all videos
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as temp_file:
        for file in crop_files:
            temp_file.write(f"file '{os.path.abspath(file)}'\n")
        temp_filename = temp_file.name

    try:
        # Use ffmpeg to concatenate all videos
        command = (
            f"ffmpeg -y -f concat -safe 0 -i {temp_filename} -c copy {output_file}"
        )

        print(f"Merging {len(crop_files)} video files...")
        subprocess.call(command, shell=True)
        print(f"Merged video saved to: {output_file}")

        return output_file

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


# ========== ========== ========== ==========
# # EXECUTE DEMO
# ========== ========== ========== ==========

# Clean up existing directories
for dir_path in [opt.work_dir, opt.crop_dir, opt.avi_dir, opt.frames_dir, opt.tmp_dir]:
    if os.path.exists(os.path.join(dir_path, opt.reference)):
        rmtree(os.path.join(dir_path, opt.reference))

# Split video into chunks
chunk_files = split_video_into_chunks(opt)

# Process each chunk
all_vidtracks = []
for chunk_idx, chunk_file in enumerate(chunk_files):
    print(f"Processing chunk {chunk_idx + 1}/{len(chunk_files)}")
    vidtracks = process_chunk(opt, chunk_file, chunk_idx)
    all_vidtracks.extend(vidtracks)

# Save results
savepath = os.path.join(opt.work_dir, opt.reference, "tracks.pckl")
with open(savepath, "wb") as fil:
    pickle.dump(all_vidtracks, fil)

merge_face_tracks(opt)

# Clean up temporary directories
rmtree(os.path.join(opt.tmp_dir, opt.reference))
