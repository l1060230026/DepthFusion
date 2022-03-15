import argparse
import errno
import struct
import gc
import glob
import os
import re
import sys

import open3d as o3d
# import matplotlib.pyplot as plt
from PIL import Image

from warp_func import *
from Cameras import Cameras

parser = argparse.ArgumentParser(description='Depth fusion with consistency check.')
parser.add_argument('--root_path', type=str, default='./')
parser.add_argument('--save_path', type=str, default='./points')
parser.add_argument('--dist_thresh', type=float, default=0.001)
parser.add_argument('--prob_thresh', type=float, default=0.6)
parser.add_argument('--num_consist', type=int, default=10)
parser.add_argument('--device', type=str, default='cpu')

args = parser.parse_args()


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python ≥ 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def read_cameras_binary(path_to_model_file):
    cameras = []
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            params = read_next_bytes(fid, num_bytes=8 * 4,
                                     format_char_sequence="d" * 4)
            cameras.append(np.array([[params[0], 0, params[2]],
                                     [0, params[1], params[3]],
                                     [0, 0, 1]]))

    return cameras


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def read_images_binary(path_to_model_file):
    rotation = []
    tranlation = []
    images_name = []
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")

            rotation.append(qvec2rotmat(np.array(binary_image_properties[1:5])))
            tranlation.append(np.array(binary_image_properties[5:8]).reshape(3, 1))
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            read_next_bytes(fid, num_bytes=24 * num_points2D,
                            format_char_sequence="ddq" * num_points2D)
            images_name.append(image_name)
    return rotation, tranlation, images_name


def read_data(path):
    path = os.path.join(path, 'sparse')
    cameras = read_cameras_binary(os.path.join(path, "cameras.bin"))
    rotation, tranlation, images_name = read_images_binary(os.path.join(path, "images.bin"))
    cameras_matrix = Cameras(cameras, rotation, tranlation, images_name)
    return cameras_matrix


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def write_pfm(file, image, scale=1):
    file = open(file, 'wb')
    color = None
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode() if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image_string = image.tostring()
    file.write(image_string)
    file.close()


def write_ply(file, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(points[:, 3:] / 255.)
    o3d.io.write_point_cloud(file, pcd, write_ascii=False)


def filter_depth(ref_depth, src_depths, ref_proj, src_projs):
    '''
	
	:param ref_depth: (1, 1, H, W)
	:param src_depths: (B, 1, H, W)
	:param ref_proj: (1, 4, 4)
	:param src_proj: (B, 4, 4)
	:return: ref_pc: (1, 3, H, W), aligned_pcs: (B, 3, H, W), dist: (B, 1, H, W)
	'''
    if args.device == 'cuda' and torch.cuda.is_available():
        ref_depth = ref_depth.cuda()
        src_depths = src_depths.cuda()
        ref_proj = ref_proj.cuda()
        src_projs = src_projs.cuda()

    ref_pc = generate_points_from_depth(ref_depth, ref_proj)
    src_pcs = generate_points_from_depth(src_depths, src_projs)

    aligned_pcs = homo_warping(src_pcs, src_projs, ref_proj, ref_depth)

    # _, axs = plt.subplots(3, 4)
    # for i in range(3):
    # 	axs[i, 0].imshow(src_pcs[0, i], vmin=0, vmax=1)
    # 	axs[i, 1].imshow(aligned_pcs[0, i],  vmin=0, vmax=1)
    # 	axs[i, 2].imshow(ref_pc[0, i],  vmin=0, vmax=1)
    # 	axs[i, 3].imshow(ref_pc[0, i] - aligned_pcs[0, i], vmin=-0.5, vmax=0.5, cmap='coolwarm')
    # plt.show()

    x_2 = (ref_pc[:, 0] - aligned_pcs[:, 0]) ** 2
    y_2 = (ref_pc[:, 1] - aligned_pcs[:, 1]) ** 2
    z_2 = (ref_pc[:, 2] - aligned_pcs[:, 2]) ** 2
    dist = torch.sqrt(x_2 + y_2 + z_2).unsqueeze(1)

    return ref_pc, aligned_pcs, dist


def parse_cameras(path):
    cam_txt = open(path).readlines()
    f = lambda xs: list(map(lambda x: list(map(float, x.strip().split())), xs))

    extr_mat = f(cam_txt[1:5])
    intr_mat = f(cam_txt[7:10])

    extr_mat = np.array(extr_mat, np.float32)
    intr_mat = np.array(intr_mat, np.float32)

    return extr_mat, intr_mat


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def load_data(cam_intr):
    '''
    :param root_path:
    :param scene_name:
    :param thresh:
    :return: depth
    '''

    depths = []
    projs = []
    rgbs = []
    w_max = 0
    h_max = 0
    for i in range(cam_intr.length):
        projs.append(torch.from_numpy(np.r_[cam_intr.matrix[i], np.array([[0, 0, 0, 1]])]))
        dep_map = read_array(os.path.join('stereo', 'depth_maps',
                                          cam_intr.images_name[i] + '.geometric.bin'))
        h_max = max(h_max, len(dep_map))
        w_max = max(w_max, len(dep_map[0]))

        rgb = np.array(Image.open(os.path.join("images", cam_intr.images_name[i])))
        rgbs.append(rgb)

    for i in range(cam_intr.length):
        dep_map = read_array(os.path.join('stereo', 'depth_maps',
                                          cam_intr.images_name[i] + '.geometric.bin'))
        dep_map = np.c_[dep_map, np.zeros((len(dep_map), w_max - len(dep_map[0])))]
        dep_map = np.r_[dep_map, np.zeros((h_max - len(dep_map), len(dep_map[0])))]
        depths.append(torch.from_numpy(np.copy(dep_map)).unsqueeze(0))
    depths = torch.stack(depths).float()
    projs = torch.stack(projs).float()
    # if args.device == 'cuda' and torch.cuda.is_available():
    #     depths = depths.cuda()
    #     projs = projs.cuda()

    return depths, projs, rgbs


def extract_points(pc, mask, rgb):
    pc = pc.cpu().numpy()
    mask = mask.cpu().numpy()

    mask = np.reshape(mask, (-1,))
    pc = np.reshape(pc, (-1, 3))
    rgb = np.reshape(rgb, (-1, 3))

    points = pc[np.where(mask)]
    colors = rgb[np.where(mask)]

    points_with_color = np.concatenate([points, colors], axis=1)

    return points_with_color


def main():
    mkdir_p(args.save_path)
    cam_intr = read_data(args.root_path)

    # mkdir_p('{}/{}'.format(args.save_path, scene))
    depths, projs, rgbs = load_data(cam_intr)
    tot_frame = depths.shape[0]
    height, width = depths.shape[2], depths.shape[3]
    points = []
    index = []
    print('total: {} frames'.format(tot_frame))
    batch_size = 5
    for i in range(tot_frame - args.num_consist):
        pc_buff = torch.zeros((3, height, width), device=args.device, dtype=depths.dtype)
        val_cnt = torch.zeros((1, height, width), device=args.device, dtype=depths.dtype)
        j = i
        mask_all = []
        while True:
            ref_pc, pcs, dist = filter_depth(ref_depth=depths[i:i + 1],
                                             src_depths=depths[j:min(j + batch_size, tot_frame)],
                                             ref_proj=projs[i:i + 1],
                                             src_projs=projs[j:min(j + batch_size, tot_frame)])

            masks = (dist < args.dist_thresh).float()
            masked_pc = pcs * masks
            pc_buff += masked_pc.sum(dim=0, keepdim=False)
            val_cnt += masks.sum(dim=0, keepdim=False)
            if not len(mask_all):
                mask_all = masks.view(batch_size, height, width).cpu().numpy().astype(np.int32)
            else:
                mask_all = np.r_[mask_all, masks.view(len(masks), height, width).cpu().numpy().astype(np.int32)]

            j += batch_size
            if j >= tot_frame:
                break
        final_mask = (val_cnt >= args.num_consist).squeeze(0)
        avg_points = torch.div(pc_buff, val_cnt).permute(1, 2, 0)

        final_pc, index_ = extract_points(avg_points, final_mask, rgbs[i], mask_all)

        points.append(final_pc)
        index.append(index_)
        torch.cuda.empty_cache()
        print('Processing  {}/{} ...'.format(i + 1, tot_frame))

    for i in range(len(points)):
        with open("numpy/" + str(i) + ".txt", "w") as f:
            for j in range(len(points[i])):
                f.write(str(points[i][j][:3]).replace('[', '').replace(']', '') + " ")
                f.write(str(index[i][j]).replace('[', '').replace(']', '').replace(',', '') + "\n")
    write_ply('{}/{}.ply'.format(args.save_path, 'scene'), np.concatenate(points, axis=0))
    del points, depths, rgbs, projs

    gc.collect()
    print('Save {}/{}.ply successful.'.format(args.save_path, 'scene'))


def merge(root_path, ):
    all_scenes = open(args.data_list, 'r').readlines()
    all_scenes = list(map(str.strip, all_scenes))
    for scene in all_scenes:
        mkdir_p('{}/{}'.format(args.save_path, scene))
        points = []
        paths = sorted(glob.glob('{}/{}/*.npy'.format(root_path, scene, )))
        for p in paths:
            points.append(np.load(p))
        points = np.concatenate(points, axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(points[:, 3:] / 255.)

        o3d.io.write_point_cloud("{}/{}.ply".format(args.save_path, scene), pcd, write_ascii=False)
        print('Save {}/{}.ply successful.'.format(args.save_path, scene))


if __name__ == '__main__':
    with torch.no_grad():
        main()
