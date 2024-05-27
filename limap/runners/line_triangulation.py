import os
import numpy as np
from tqdm import tqdm
global view

import cv2
import limap.visualize
import limap.base as _base          # 基本操作
import limap.merging as _mrg        # 数据融合
import limap.triangulation as _tri  # 三角测量
import limap.vplib as _vplib        # 视点处理？
import limap.pointsfm as _psfm      # 点云生成
import limap.optimize as _optim     # 优化
import limap.runners as _runners    # 运行器
import limap.util.io as limapio     # io输入输出操作
import limap.visualize as limapvis  # 可视化

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    custom_data = param[0]
    image_path = custom_data['image_path']
    if event == cv2.EVENT_LBUTTONDOWN:
         # step 0 点击选中的像素
        xy = "%f,%f" % (x, y)
        print(f"-----------------------------------------------------------------")
        print(f"LBUTTONDOWN sellect:{x} {y}")

        # step 1  图上绘制
        # 在图片上描点的时候，对关键点（point2D_X，point2D_Y）四舍五入然后描点，要不然没有办法描小数
        cv2.circle(img, (round(x), round(y)), 1, (0, 0, 255), thickness=-1)  
        point2d_xy = "point2d:%f %f" % (x, y)
        cv2.putText(img, point2d_xy, (round(x)-100, round(y)-8), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)

# input参数：配置、图像集合、邻居图像(依靠colmap的三角测量共视信息)
# output  ：输出的3D线条轨迹列表
def line_triangulation(cfg, imagecols, neighbors=None, ranges=None):
    '''
    Main interface of line triangulation over multi-view images.

    Args:
        cfg (dict): Configuration. Fields refer to :file:`cfgs/triangulation/default.yaml` as an example
        imagecols (:class:`limap.base.ImageCollection`): The image collection corresponding to all the images of interest
        neighbors (dict[int -> list[int]], optional): visual neighbors for each image. By default we compute neighbor information from the covisibility of COLMAP triangulation.
        ranges (pair of :class:`np.array` each of shape (3,), optional): robust 3D ranges for the scene. By default we compute range information from the COLMAP triangulation.
    Returns:
        list[:class:`limap.base.LineTrack`]: list of output 3D line tracks
    '''
    # step 0 设置运行配置，图像预处理
    print("[LOG] Number of images: {0}".format(imagecols.NumImages()))
        # step 0.1 设置运行配置
    cfg = _runners.setup(cfg)
    detector_name = cfg["line2d"]["detector"]["method"]
    if cfg["triangulation"]["var2d"] == -1:
        cfg["triangulation"]["var2d"] = cfg["var2d"][detector_name]
        # step 0.2 图像预处理（去畸变、调整大小）
    # undistort images
    if not imagecols.IsUndistorted():
        imagecols = _runners.undistort_images(imagecols, os.path.join(cfg["dir_save"], cfg["undistortion_output_dir"]), skip_exists=cfg["load_undistort"] or cfg["skip_exists"], n_jobs=cfg["n_jobs"])
    # resize cameras
    assert imagecols.IsUndistorted() == True
    if cfg["max_image_dim"] != -1 and cfg["max_image_dim"] is not None:
        imagecols.set_max_image_dim(cfg["max_image_dim"])
    limapio.save_txt_imname_dict(os.path.join(cfg["dir_save"], 'image_list.txt'), imagecols.get_image_name_dict())
    limapio.save_npy(os.path.join(cfg["dir_save"], 'imagecols.npy'), imagecols.as_dict())

    # step 1 【点特征-colmap】通过sfm，从多视图图像中提取元信息(邻居图像neighbors/ 三维重建范围ranges)
    ##########################################################
    # [A] sfm metainfos (neighbors, ranges)
    ##########################################################
    sfminfos_colmap_folder = None # 用于存储COLMAP的元信息文件夹路径
    # 如果没有邻居视图的先验信息，表明要计算邻居视图信息和3D范围
    if neighbors is None:
        sfminfos_colmap_folder, neighbors, ranges = _runners.compute_sfminfos(cfg, imagecols) # NOTICE 涉及到colmap的主函数：点特征提取&匹配 ， 三角化（有先验位姿） / mapping
    else:
        limapio.save_txt_metainfos(os.path.join(cfg["dir_save"], "metainfos.txt"), neighbors, ranges)
        neighbors = imagecols.update_neighbors(neighbors)
        for img_id, neighbor in neighbors.items():
            neighbors[img_id] = neighbors[img_id][:cfg["n_neighbors"]]
    limapio.save_txt_metainfos(os.path.join(cfg["dir_save"], "metainfos.txt"), neighbors, ranges)

    # step 2 【线特征提取】对于每张图像，提取其中的2D线段，并计算描述符用于后面的匹配 
    ##########################################################
    # [B] get 2D line segments for each image
    ##########################################################
    # step 2.1 根据配置判断是否要计算描述符 compute_descinfo
    # os 描述符的目的是捕捉线段的关键信息，使得它们可以在不同的图像之间进行比对和识别。
            # 2.1.1  根据配置中“是否用穷尽匹配”，来判断是否需要计算描述符compute_descinfo （如果不用穷尽匹配，则需要描述符）
    compute_descinfo = (not cfg["triangulation"]["use_exhaustive_matcher"]) # 这个设置是因为穷尽匹配器通常会考虑所有可能的图像对，可能不需要预先计算描述符信息。
            # 2.2.2  （不用穷尽匹配 + 不加载预先计算的匹配 + 不加载检测器输出det） || 指定要计算描述符
    compute_descinfo = (compute_descinfo and (not cfg["load_match"]) and (not cfg["load_det"])) or cfg["line2d"]["compute_descinfo"]
    # step 2.2 提取二维线段all_2d_segs 计算线段的描述符信息
    all_2d_segs, descinfo_folder = _runners.compute_2d_segs(cfg, imagecols, compute_descinfo=compute_descinfo) # NOTICE 2D线特征提取

    # step 3 【线特征匹配】
    ##########################################################
    # [C] get line matches
    ##########################################################
    # 如果配置中不用“详尽匹配exhaustive_matcher”，才执行此步
    # [LOG] Start matching 2D lines...
    if not cfg["triangulation"]["use_exhaustive_matcher"]:
        matches_dir = _runners.compute_matches(cfg, descinfo_folder, imagecols.get_img_ids(), neighbors)

    # step 4 【多视图三角化】- 从二维线段数据中构建三维线段
    ##########################################################
    # [D] multi-view triangulation
    ##########################################################
        # step 4.1 初始三角化模块 Triangulator
    print( f"step 4.1 \n")
    Triangulator = _tri.GlobalLineTriangulator(cfg["triangulation"]) # 依据cfg["triangulation"]初始化，用于后续的线段三角化任务
        # step 4.2 设置三维空间范围（从点重建中得到的）
    print( f"step 4.2 \n")
        # os 用于约束线段的三维位置，使得三角化结果更准确
    Triangulator.SetRanges(ranges)
        # step 4.3 初始化三角化数据
    print( f"step 4.3 \n")
    all_2d_lines = _base.get_all_lines_2d(all_2d_segs)
    Triangulator.Init(all_2d_lines, imagecols)# 用所有图像集合和提取的2D线段初始化
        # step 4.4 处理灭点
        # 如果启用灭点处理，则创建一个灭点检测器，用于检测所有图像的灭点
    print( f"step 4.4 \n")
    if cfg["triangulation"]["use_vp"]:
        vpdetector = _vplib.get_vp_detector(cfg["triangulation"]["vpdet_config"], n_jobs = cfg["triangulation"]["vpdet_config"]["n_jobs"])
        vpresults = vpdetector.detect_vp_all_images(all_2d_lines, imagecols.get_map_camviews())
        Triangulator.InitVPResults(vpresults) # 灭点检测结果用于初始化三角化处理器，以便在三角化时考虑灭点信息，可能有助于深度估计等
        # step 4.5 使用COLMAP重建数据（可以重用sfminfos_colmap中的模型）
    # get 2d bipartites from pointsfm model
    print( f"step 4.5 \n")
    if cfg["triangulation"]["use_pointsfm"]["enable"]:                      # 如果使用点重建
        # OS 根据配置情况，重新运行COLMAP或使用预先存在的COLMAP模型。
        # os A. 没有预先加载的colmap结果，则重新运行colmap
        if cfg["triangulation"]["use_pointsfm"]["colmap_folder"] is None:   
            colmap_model_path = None
            # check if colmap model exists from sfminfos computation       
            # os a. 如果设置重用sfminfos_colmap模型，并且sfminfos_colmap_folder非空： 则直接使用该模型路径
            if cfg["triangulation"]["use_pointsfm"]["reuse_sfminfos_colmap"] and sfminfos_colmap_folder is not None:
                colmap_model_path = os.path.join(sfminfos_colmap_folder, "sparse")
                if not _psfm.check_exists_colmap_model(colmap_model_path):
                    colmap_model_path = None
            # retriangulate
            # os b. 如果不存在可重用的colmap模型：会在指定目录下重新运行 COLMAP，将结果存储在 colmap_output_path 中，并设置 colmap_model_path
            if colmap_model_path is None:
                colmap_output_path = os.path.join(cfg["dir_save"], "colmap_outputs_junctions")
                input_neighbors = None
                if cfg["triangulation"]["use_pointsfm"]["use_neighbors"]:
                    input_neighbors = neighbors
                _psfm.run_colmap_sfm_with_known_poses(cfg["sfm"], imagecols, output_path=colmap_output_path, skip_exists=cfg["skip_exists"], neighbors=input_neighbors)
                colmap_model_path = os.path.join(colmap_output_path, "sparse")
        # os B. 如果提供了预先加载的 COLMAP 结果路径，则直接使用该路径 
        # NOTICE
        else:
            print(f"step 4.6 \n")
            colmap_model_path = cfg["triangulation"]["use_pointsfm"]["colmap_folder"]
        # OS 使用 PyReadCOLMAP 读取 COLMAP 的重建结果。
        reconstruction = _psfm.PyReadCOLMAP(colmap_model_path)
        # OS 基于 COLMAP 的重建结果，计算二分图。这一步的目的是为了在二维线段和三维点之间建立对应关系
        all_bpt2ds, sfm_points = _runners.compute_2d_bipartites_from_colmap(reconstruction, imagecols, all_2d_lines, cfg["structures"]["bpt2d"]) # NOTICE
        # OS 将计算得到的二分图设置给三维线段的三角测量器。
        Triangulator.SetBipartites2d(all_bpt2ds)
        # OS 如果配置中指定要使用 COLMAP 三维重建结果中的三角化点,将 COLMAP 的三角化点设置给三维线段的三角测量器。
        if cfg["triangulation"]["use_pointsfm"]["use_triangulated_points"]:
            Triangulator.SetSfMPoints(sfm_points)
    # triangulate
        # step 4.6 进行三角化
        # 对每个图像进行三角化处理，根据图像的匹配关系，确定每条线段的三维坐标
    print('Start multi-view triangulation...')
    for img_id in tqdm(imagecols.get_img_ids()):
        if cfg["triangulation"]["use_exhaustive_matcher"]:  # 如果使用详尽匹配
            Triangulator.TriangulateImageExhaustiveMatch(img_id, neighbors[img_id])
        else:
            matches = limapio.read_npy(os.path.join(matches_dir, "matches_{0}.npy".format(img_id))).item()
            Triangulator.TriangulateImage(img_id, matches)
    linetracks = Triangulator.ComputeLineTracks()  # 从所有三角化的结果中生成三维线段跟踪信息。

    # filtering 2d supports
        # step 4.7 过滤和重组线段
    # filtertracksbyreprojection 移除重投影误差大于阈值的线段
    linetracks = _mrg.filtertracksbyreprojection(linetracks, imagecols, cfg["triangulation"]["filtering2d"]["th_angular_2d"], cfg["triangulation"]["filtering2d"]["th_perp_2d"])
    if not cfg["triangulation"]["remerging"]["disable"]:
        # remerging
        linker3d = _base.LineLinker3d(cfg["triangulation"]["remerging"]["linker3d"])
        linetracks = _mrg.remerge(linker3d, linetracks) # 再次重组线段以提高整体的准确性和一致性
        linetracks = _mrg.filtertracksbyreprojection(linetracks, imagecols, cfg["triangulation"]["filtering2d"]["th_angular_2d"], cfg["triangulation"]["filtering2d"]["th_perp_2d"])
    # 根据灵敏度和重叠度过滤线段
    linetracks = _mrg.filtertracksbysensitivity(linetracks, imagecols, cfg["triangulation"]["filtering2d"]["th_sv_angular_3d"], cfg["triangulation"]["filtering2d"]["th_sv_num_supports"])
    linetracks = _mrg.filtertracksbyoverlap(linetracks, imagecols, cfg["triangulation"]["filtering2d"]["th_overlap"], cfg["triangulation"]["filtering2d"]["th_overlap_num_supports"])
    # 筛选那些在足够多的图像中的共视线段，确保结果的可靠性和稳定性
    validtracks = [track for track in linetracks if track.count_images() >= cfg["n_visible_views"]]
    
    # TODO 这里加交互界面 linetracks
    # 1. 访问linetracks，让他们可视化
        # 初始化相机视图
    cam1,final_pose = _psfm.Readcam(cfg["colmap_path"])
    # print(f"!!!Result(P+L) Pose (qvec, tvec): {final_pose.qvec}, {final_pose.tvec}\n") # 检验一下是不是最后一张图的
    camview = _base.CameraView(cam1, final_pose)
    # img = camview.read_image(set_gray=False)

    final_img_id = imagecols.get_img_ids()[-1]
    print(f"final_img_id:{final_img_id}") # 检验一下是不是最后一张图的

    # camview = imagecols.camview(final_img_id)
    image_path = imagecols.image_name(final_img_id)
    # img = camview.read_image(set_gray = False)
    # img = imagecols.read_image(final_img_id, set_gray=False)

    print(f"image_path = {image_path}")
    img = cv2.imread(image_path)
    cv2.imshow("img", img)
    cv2.waitKey(0)           # 【等待按键后】关闭窗口
    cv2.destroyAllWindows()  # 关闭所有 OpenCV 创建的窗口
    
    # img = limap.visualize.draw_segments(img,all_2d_segs,(0,255,0))
    for linetrack in linetracks:
        if final_img_id in linetrack.image_id_list:
            # 获取当前图像在LineTrack中的索引
            index = linetrack.image_id_list.index(final_img_id)
            # 获取对应的2D线段
            l2d = linetrack.line2d_list[index]
            l3d = linetrack.line
            l2d_proj = l3d.projection(camview)
            img = cv2.line(img, l2d_proj.start.astype(int), l2d_proj.end.astype(int), color=[0, 0, 255])
    
    cv2.imshow("img_segs", img)
    cv2.waitKey(0)           # 【等待按键后】关闭窗口
    cv2.destroyAllWindows()  # 关闭所有 OpenCV 创建的窗口
    cv2.imwrite((cfg["output_dir"] / "img_segs.png").as_posix(), img)

    # 2. 通过鼠标点击，确定结构性线条
    # 3. 对结构性线条的linetracks.is_horizontal/linetracks.is_plumb进行修改

    # step 5 几何优化 结合点线特征，通过BA来优化视觉重建中相机姿态和线条轨迹
    # IDEA 这里可以像colmap里面一样改一下BA
    # TODO 整理优化的思路
    ##########################################################
    # [E] geometric refinement
    ##########################################################
    if not cfg["refinement"]["disable"]: # 启用优化
        cfg_ba = _optim.HybridBAConfig(cfg["refinement"])    # 创建一个点线混合优化对象， 并从cfg["refinement"]中加载配置参数
        cfg_ba.set_constant_camera()                         # 设置相机的内外参为常量，不参与优化 # ？（是害怕线特征会把pose优化不好吗）
        ba_engine = _optim.solve_line_bundle_adjustment(cfg["refinement"], imagecols, linetracks, max_num_iterations=200) # NOTICE 优化主函数
        linetracks_map = ba_engine.GetOutputLineTracks(num_outliers=cfg["refinement"]["num_outliers_aggregator"])         # 获取优化后的线段轨迹
        linetracks = [track for (track_id, track) in linetracks_map.items()]    # 将优化后的linetracks由字典结构转为列表结构

        pointtracks_map = ba_engine.GetOutputPointTracks()
        pointtracks = [track for (track_id, track) in pointtracks_map.items()]    # 将优化后的linetracks由字典结构转为列表结构


    # step 6 输出及可视化
    ##########################################################
    # [F] output and visualization
    ##########################################################
    # save tracks
        # 将线段轨迹以文本格式保存在指定的目录。n_visible_views=4 表示只保存那些在至少4个视图中可见的轨迹。
    limapio.save_txt_linetracks(os.path.join(cfg["dir_save"], "alltracks.txt"), linetracks, n_visible_views=4) 
    # TODO 待修改↓存储点特征的代码
    # limapio.save_txt_pointtracks(os.path.join(cfg["dir_save"], "pointtracks.txt"), pointtracks_map)

        #  将线段轨迹以及关联的配置信息、图像集和二维线段数据保存在特定的目录下，便于后续分析和验证。
    limapio.save_folder_linetracks_with_info(os.path.join(cfg["dir_save"], cfg["output_folder"]), linetracks, config=cfg, imagecols=imagecols, all_2d_segs=all_2d_segs)
        # 以OBJ形式保存三维线段
    VisTrack = limapvis.Open3DTrackVisualizer(linetracks) # 初始化可视器
    VisTrack.report()
    limapio.save_obj(os.path.join(cfg["dir_save"], 'triangulated_lines_nv{0}.obj'.format(cfg["n_visible_views"])), VisTrack.get_lines_np(n_visible_views=cfg["n_visible_views"]))

    # visualize
    if cfg["visualize"]:
            # 筛选出在足够多视图中可见的轨迹
        validtracks = [track for track in linetracks if track.count_images() >= cfg["n_visible_views"]] 
            # 通过定义的report_track函数可以单独查看每条轨迹
        def report_track(track_id):
            limapvis.visualize_line_track(imagecols, validtracks[track_id], prefix="track.{0}".format(track_id))
        import pdb
        pdb.set_trace()
        VisTrack.vis_reconstruction(imagecols, n_visible_views=cfg["n_visible_views"], width=2)
        pdb.set_trace()
    return linetracks

