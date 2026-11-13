# Baxter 与 D455 AprilTag 标定操作与交接手册

更新日期：2026 年 11 月 13 日。对象：负责打印、安装、采样、核验及交付配置的现场同学。

本手册针对 D455 固定在机器人外部、Baxter 末端携带标签的 eye-on-base 标定。目标是获得可靠的相机到机器人 base 的外参，并完成工具 TCP 和桌面几何的独立核验。正文中的示例尺寸、frame 名称和话题名称必须与现场实测一致；本文没有连接现场设备确认其当前运行状态，也没有执行标定。

## 1 这次必须完成什么

最终交付不只是一个 YAML 文件，而是以下内容：

1. 纸质标签的家族、ID、横纵实测尺寸和安装照片。
2. 相机序列号、颜色流分辨率、CameraInfo、实际 optical frame。
3. 机器人 base 与选定末端 frame，标签与末端刚性连接的说明。
4. 多姿态原始样本、求解结果、未参与求解的验证记录。
5. 明确方向的 camera optical → Baxter base 变换。
6. 左右夹爪各自的 TCP 测量结果或尚未完成的明确标记。
7. 更新后的项目本地配置，以及只读定位与 IK 检查记录。

“软件算出了数值”“RViz 能显示标签”和“标定通过”是三个不同层级。没有完成独立验证，不要将配置标记为已标定并直接下探抓物。

## 2 现场背景与需要纠正的问题

| 设备 | 历史地址或目录 | 用途 |
| --- | --- | --- |
| Baxter 控制主机 | `cothink@192.168.1.101` | ROS、D455、标签检测、标定计算 |
| Baxter 本体 | `192.168.1.100` | 机器人 ROS master |
| 天数板子 | `cix@192.168.1.103` | 模型和任务循环；标定本身不需要 VLM/VLA |
| 控制主机项目目录 | `/home/cothink/cix-harness-lab-20261106` | 现场源码与本地配置 |
| 板端项目目录 | `/home/cix/cix-harness-lab-20261106` | 模型服务和任务入口 |

地址和目录需要重新确认。密码、私钥不写入本手册。

旧标定文件曾位于 `~/.ros/easy_handeye/baxter_d455_handeye_calibration_eye_on_base.yaml`。最近一次核验中，旧外参把桌面映射到 base z≈−0.047 m，而用户提供的桌面上表面约为 −0.18 m，存在约 13 cm 的差异。因此旧文件只能保留作对照，不能直接继续当作精确标定。

此前原语抓取成功使用的是人工对准物体得到的局部参考，包含工具偏移估计，不是完整外参标定。新标定需要替换这种局部假设，同时保留旧记录以便追溯。

机器人自身左手是 `left`，可能出现在相机画面右侧。所有记录明确区分“机器人左/右臂”和“画面左/右”。

## 3 标定原理与坐标约定

本次相机固定不动，标签固定在末端并随手臂运动。每个样本包含：

```text
机器人给出：base ← end_effector
检测器给出：camera_optical ← tag
待求结果：  base ← camera_optical
```

定义 `T_A_B` 为把 B 坐标中的点变换到 A 坐标的矩阵。每组样本应满足：

```text
T_base_effector(i) × T_effector_tag
    = T_base_camera × T_camera_tag(i)
```

其中 `T_effector_tag` 未知但固定。多组不同姿态使求解器能够求出相机外参，不需要提前精确测量标签相对末端的安装位姿。[easy_handeye 模式说明](https://github.com/IFL-CAMP/easy_handeye)

不能把标签随意放在桌面上、仅检测一次就得到 camera→base，除非另有精确测得的标签 base 位姿，并使用相应的已知参考标定方法。

外参、相机内参和 TCP 必须分开处理：

| 项目 | 作用 | 本次做法 |
| --- | --- | --- |
| 内参和畸变 | 像素到光线、标签姿态估计 | 读取匹配图像流的 CameraInfo，先检查合理性 |
| 相机外参 | 相机三维点转换到 base | AprilTag 多姿态手眼标定 |
| 工具 TCP | SDK 末端点转换到实际夹持参考点 | 单独测量左右工具；不由手眼标定自动得到 |
| 桌面与物体几何 | 下探与释放高度 | 在新外参、正确 TCP 基础上实测核验 |

## 4 标签选型与材料

起始建议：`tag36h11`，ID 0，黑框外边长 100 mm，四周保留完整白边。这个尺寸适合先验证，若实际距离下标签成像太小再增大，不能只靠软件放大图像补偿。

材料：哑光白纸、高质量黑白打印、平整硬背板、双面胶或固定胶、刚性支架、钢尺或卡尺。约 140×140 mm 背板可作为起点，以实际安装空间为准。

避免软纸板弯曲、标签表面反光塑封、透明胶穿过黑白图案和紧贴黑框裁剪。背板不能在转动中晃动或遮挡关节、线缆。

从 [AprilRobotics 官方图案库](https://github.com/AprilRobotics/apriltag-imgs) 的 `tag36h11` 目录下载 ID 0 原文件，通常名为 `tag36_11_00000.png`。不要使用网页截图、AI 生成图案或自行猜测黑白格。

本次选择 AprilTag 不代表控制主机旧识别节点已经兼容。ArUco 与 AprilTag 不是同一种编码；需要确认检测器和家族参数，easy_handeye 只负责标定计算。

## 5 打印与尺寸验收

### 5.1 尺寸指哪里

对于 tag36h11，检测尺寸是黑框外边长，不包含外侧白边。一般定义为检测角点之间的边长。[AprilTag 官方定义](https://github.com/AprilRobotics/apriltag/blob/master/README.md)

```text
完整图片：白边 + 黑框/编码区 + 白边
size：   只测黑框外边到对侧黑框外边
```

若原图为 10×10 格，黑框外沿覆盖其中 8×8 格，则希望黑框为 100 mm 时，应将完整图像宽高设为 125 mm。先确认图案实际边界，额外加白边或裁剪后的图片不能直接套用这个比例。

### 5.2 排版与打印

1. 在 A4 页面插入原图，保持宽高比为 1。
2. 按黑框和完整图片的比例设置物理尺寸。
3. 原图像素较少时采用最近邻放大或忠实转换为矢量方格，避免平滑插值产生模糊灰边。
4. 在白色保护区之外标注家族、ID 和标称尺寸。
5. 导出 PDF，放大检查边缘和方格。
6. 打印选择“实际大小/100%”，关闭适合页面、自动缩放和省墨模式，优先 600 dpi 或更高质量。
7. 不要用手机照片再打印，也不要在不同软件中反复缩放。

打印后必须实测，打印机可能改变目标尺寸。[Kalibr 打印提醒](https://github.com/ethz-asl/kalibr/wiki/calibration-targets)

### 5.3 记录实际尺寸

| 字段 | 现场填写 |
| --- | --- |
| 家族 / ID | tag36h11 / 0 |
| 黑框水平边长 | ____ mm |
| 黑框竖直边长 | ____ mm |
| 四边及方正性 | ____ |
| 检测器采用边长 | ____ m |
| 打印日期与文件 | ____ |

例如黑框实测 99.6 mm，程序尺寸填写 0.0996，而不是照填 0.1000。如果水平和竖直明显不一致、边缘不直或测量差异超过量具重复误差，应重新打印；不要靠平均数掩盖非等比缩放。

## 6 末端安装与采样前准备

先正常安放当前夹持物，再安装标定板。优先固定在末端刚性部件或专用支架，不贴在会随开合移动的单根夹指上。若只能夹持背板，必须确保整个采样过程不滑动、不开合、不改变夹持位置。

标签不要求与末端坐标轴平行，也不要求安装在中心；要求是相对记录的末端 frame 固定，并在多个姿态下可见。标签到末端偏移无需精确测量，但安装照片和固定方式要留下记录。

固定 D455 支架，整理线缆。采样中只改变机械臂姿态；相机移动、标签松动或重装都意味着数据集的固定关系改变，应重新采集。

选择只用左臂或右臂完成一组标定，不在同一数据集中切换 effector frame。得到同一相机相对同一 base 的外参后可服务双臂；左右 TCP 则分别标定。

## 7 控制主机的软件和相机检查

### 7.1 设置 ROS 环境

在控制主机终端执行，先核对工作区路径：

```bash
source /opt/ros/noetic/setup.bash
source /home/cothink/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.100:11311
export ROS_IP=192.168.1.101
unset ROS_HOSTNAME
rostopic list
rospack find easy_handeye
rospack find apriltag_ros
rospack find realsense2_camera
rospack find image_proc
```

缺包时先检查现有工作区和对应 ROS Noetic 包，不将 ROS 2 包装进 ROS 1 环境。不要为这次标定随意升级现有驱动、COREX 或整个 Python 环境。

### 7.2 处理 D455 独占

当前项目桥直接通过 pyrealsense2 使用 D455。启用 ROS 相机驱动前，正常停止当前占用相机的桥，等待设备释放。只保留一个相机驱动进程；不同时启动两个 bridge 或两个 RealSense 驱动。

标定采样时停止自主任务循环和自动运动。Baxter 的基础 ROS 状态发布继续运行；不要停止整个机器人系统。采样由现场人员按既有拖动操作改变姿态。

### 7.3 启动相机并核对话题

采用控制主机已验证的 RealSense 启动方式。如下仅为常见 ROS 1 启动模板，参数必须与现场安装版本支持的选项核对：

```bash
roslaunch realsense2_camera rs_camera.launch \
  color_width:=640 color_height:=480 color_fps:=30
```

查看实际话题：

```bash
rostopic list | grep -E 'camera_info|image_raw|image_rect'
rostopic echo -n 1 /camera/color/camera_info
rostopic hz /camera/color/image_raw
```

后两行以 `/camera/color` 为示例。记录实际命名空间和 CameraInfo 的 `header.frame_id`。

apriltag_ros 的示例输入是校正图 `image_rect`。如果驱动只有原图，使用匹配的 image_proc 生成校正图，不能仅把原图话题改名成 image_rect。常见 ROS 1 形式为：

```bash
ROS_NAMESPACE=/camera/color rosrun image_proc image_proc
```

再次确认生成的校正图话题，并确认其匹配的 CameraInfo。不同安装版本的话题可能是 `image_rect` 或 `image_rect_color`，下面用实际名字替换。分辨率、内参和畸变处理必须一致。

## 8 启动 AprilTag 检测

官方 `continuous_detection.launch` 会在节点内部加载 settings.yaml 和 tags.yaml，并清理旧参数。不要假设在外部提前 `rosparam set` 就一定生效。[上游启动文件](https://github.com/AprilRobotics/apriltag_ros/blob/master/apriltag_ros/launch/continuous_detection.launch)

建议在本次本地标定目录创建独立启动文件 `detect_tag.launch`，显式指定自己的标签配置。以下为模板，保存前替换实际图像话题、CameraInfo 话题和实测边长：

```xml
<launch>
  <node pkg="apriltag_ros" type="apriltag_ros_continuous_node"
        name="calibration_apriltag" output="screen">
    <remap from="image_rect" to="/camera/color/image_rect_color"/>
    <remap from="camera_info" to="/camera/color/camera_info"/>
    <rosparam command="load" file="$(find apriltag_ros)/config/settings.yaml"/>
    <param name="tag_family" value="tag36h11"/>
    <param name="publish_tf" value="true"/>
    <param name="publish_tag_detections_image" value="true"/>
    <rosparam param="standalone_tags">
      [{id: 0, size: 0.1000, name: calibration_tag}]
    </rosparam>
    <rosparam param="tag_bundles">[]</rosparam>
  </node>
</launch>
```

`0.1000` 只是示例，不是代替实测的默认结论。配置结构与尺寸单位来自 [官方标签配置](https://github.com/AprilRobotics/apriltag_ros/blob/master/apriltag_ros/config/tags.yaml)。

在保存该文件的目录执行：

```bash
roslaunch ./detect_tag.launch
rosparam get /calibration_apriltag/tag_family
rosparam get /calibration_apriltag/standalone_tags
rostopic list | grep tag
```

另开终端在图像工具或 RViz 查看检测结果，确认 ID=0、四角与黑框吻合。用实际 optical frame 检查 TF，例如：

```bash
rosrun tf tf_echo camera_color_optical_frame calibration_tag
```

如果真实 frame 不叫这个名字，必须替换。不要将 camera_link、depth optical 和 color optical 混为一谈。

### 检测进入采样前的条件

- 标签静止时能稳定识别，姿态不频繁翻转。
- 图案清晰，无明显反光、运动模糊、角点遮挡。
- 距离估计与尺量处于合理一致范围，没有明显比例错误。
- 不同采样朝向下仍有足够成像面积。
- 标签消失时不把最后一次检测误当成新样本。

检测不稳定先处理光照、尺寸、焦点和图像配置，不通过增加样本数掩盖问题。

## 9 启动手动采样标定

首先确定四个 frame，填写以下表格：

| 参数 | 候选值 | 现场确认值 |
| --- | --- | --- |
| robot_base_frame | base | ____ |
| robot_effector_frame | left_gripper | ____ |
| tracking_base_frame | camera_color_optical_frame | ____ |
| tracking_marker_frame | calibration_tag | ____ |

`left_gripper` 只是候选，必须确认与标签安装部件刚性连接，并能查询 base→effector。不要因名字相似就使用 wrist 或实际会移动的 finger frame。

示例检查：

```bash
rosrun tf tf_echo base left_gripper
rosrun tf tf_echo camera_color_optical_frame calibration_tag
```

以下启动模板在所有 frame 核对后使用：

```bash
roslaunch easy_handeye calibrate.launch \
  eye_on_hand:=false \
  namespace_prefix:=baxter_d455_apriltag_new \
  freehand_robot_movement:=true \
  robot_base_frame:=base \
  robot_effector_frame:=left_gripper \
  tracking_base_frame:=camera_color_optical_frame \
  tracking_marker_frame:=calibration_tag \
  publish_dummy:=false \
  start_rviz:=false
```

`freehand_robot_movement=true` 明确使用手动移动，避免默认启用自动运动组件。`publish_dummy=false` 避免把临时占位变换误当作真实外参；此时 TF 可以是两个独立树，只要各自的 base→effector 和 camera→tag 可查询即可。关闭 RViz 不影响采样 GUI。参数依据 [easy_handeye calibrate.launch](https://github.com/IFL-CAMP/easy_handeye/blob/master/easy_handeye/launch/calibrate.launch)。

同时检查并停止旧标定结果的静态发布节点，防止同一 camera frame 出现多个相互矛盾的父变换。不要删除旧文件，只停止不适用于本次采样的发布。

建议在控制主机本机桌面打开 GUI，避免 Windows SSH 图形转发成为额外问题。本手册不要求配置 MoveIt 自动采样。

## 10 样本采集规范

建议先采 20–30 组有效样本，另留 5–10 组独立验证姿态。这是本项目工程起点，不是算法最低样本数或精度保证。

有效姿态必须兼有平移和多轴旋转。覆盖计划抓取区的左、中、右，不同高度和远近；适当改变朝上/下、朝左/右的倾斜，始终让标签可见。不只做平移，也不只围绕同一根轴转动。

每次按以下步骤操作：

1. 手动移动到新姿态，不触碰标签背板。
2. 释放拖动操作，等待机器人与标签稳定；可先等待约 1–2 秒，再根据图像和遥测判断。
3. 确认本次检测是新数据，ID 正确、角点清晰、没有位姿跳变。
4. 在采样 GUI 中点击 Take Sample。
5. 记录样本编号、位置/朝向特点、异常情况。
6. 对误采的运动中样本、遮挡样本使用 Remove Sample，不混入求解。

不要为了多采样本强行走不可达的长竖直路径。采用当前构型可达的多方向姿态，另一只手臂保持不干涉视野与运动空间。

### 时钟与同步

本项目曾发现机器人时钟比控制主机快约 0.13 秒。项目桥中的 `robot_clock_offset_s` 是桥内部校验补偿，不会自动修复 easy_handeye 的 ROS TF 时间关系。标定前检查机器人状态、图像和 TF 时间戳；若存在明显错位，先解决时间同步或采用一致时间处理，不能直接修改标定样本掩盖它。

停稳采样能降低不同采样瞬间带来的误差，但不能代替正常的时间与数据新鲜度检查。

### 原始数据留存

建议同时录制必要 ROS bag：图像、对应 CameraInfo、`/tf`、`/tf_static`、标签检测话题，以及需要的机器人状态。话题名以 `rostopic list` 为准，先确认磁盘空间；原始图像数据量较大。

若当前 GUI 只保存最终结果，不自动导出全部原始样本，应另外保存采样服务中的数据或配套 bag 与样本编号。不要等采样窗口关闭后才发现只有一个最终矩阵。

## 11 求解与验收

完成采样后在 GUI 中 Compute，记录使用算法及有效样本数，再 Save。保存路径以工具实际输出为准，使用新的 namespace，避免覆盖旧 4 月结果。

不能根据“求解成功”直接启用抓取。先检查相机位姿是否符合物理安装，再使用留出的姿态验证：

```text
对每个验证姿态计算：
T_effector_tag(i)
  = inverse(T_base_effector(i)) × T_base_camera × T_camera_tag(i)
```

标签刚性固定时，这个结果应在不同姿态下基本一致。记录其平移与旋转的离散程度；不只记录均值，也保留最大偏差和最差姿态。异常样本应查原因，不仅为了好看而删除。

再进行外部几何验证：

1. 在桌面多个位置验证图像/深度点转换后的 base 坐标。
2. 与独立测得的参考点比较；参考点测量若依赖末端触碰，要先保证 TCP 已知，否则两种误差会混在一起。
3. 检查是否仍存在类似此前 13 cm 的系统偏差。
4. 检查误差是否随远近、图像位置或机械臂朝向变化。

目前桌面 `z≈−0.18 m` 只是粗核对范围，不应强行调平移使结果“刚好等于 −0.18”。最终桌面值应来自本次测量。

### 如何判定是否够用

验收阈值应由方块尺寸、夹爪开口与抓取余量确定。举例：开口宽 W、物体宽 B 时，理想横向余量约为 `(W−B)/2`，视觉定位、外参、TCP、机器人执行误差的合计应明显小于这个余量。

对于当前小方块任务，可先以独立点检不出现厘米级系统偏差、重复测量稳定为基本目标，并争取约 5–10 mm 范围内的点位误差；这只是初期工程目标，不是保证，也不是对所有物体统一适用的标准。若误差达到或超过可用夹持余量，应继续排查，不通过放大运动容差掩盖几何错误。

记录结论为“通过当前指定任务范围”“仅局部可用”或“未通过”，明确覆盖区域。只验证中心一点不能宣称整张桌面精确。

## 12 TCP 与桌面参数的独立核验

相机外参正确后，长夹指仍可能让实际指尖与 SDK 末端不同。项目使用：

```text
TCP_base = EEF_base + R_base_tool × tcp_offset_tool_m
```

`tcp_offset_tool_m` 必须表示在代码使用的末端姿态坐标系中的偏移，而不是直接填一个 base Z 高度差。先明确 TCP 是两指之间的实际夹持参考点还是指尖平面中心，整套配置保持同一定义。

只在一个姿态轻贴桌面，只能约束该朝向下的高度关系，不能完整确定三维工具偏移。可采用刚性参考夹具、多姿态定点/pivot 测量或可靠工具尺寸测量；方法和误差需记录。左右夹指结构及安装偏移分别测，不直接复制。

随后核验：

- 桌面上表面 `table_z_m`，在多个位置检查倾斜与起伏。
- 安全通行高度 `transit_z_m` 与实际工作区。
- 抓取下探量 `grasp_inset_m`。
- 物体底部低于 TCP 的距离 `held_bottom_below_tcp_m`。
- 放置余量 `release_clearance_m`。

当前代码用一个固定 table_z 表示桌面。如果工作区内桌面明显倾斜，单一高度不能完整描述，需限制有效范围或进一步完善平面几何，不能把问题简单归为机械臂精度。

## 13 接入项目配置

目标变换必须满足：

```text
p_base = R_camera_to_base × p_camera + t_camera_to_base
```

读取求解结果时核对 parent/child、源坐标系和目标坐标系，不只看文件名。若拿到相反方向：

```text
R_inverse = R_transpose
t_inverse = −R_transpose × t
```

不是只把平移取负。项目四元数顺序为 `[x,y,z,w]`，平移单位米。

写入 `config/baxter-primitives.local.json` 中的以下字段，保留本次元数据和旧配置备份：

| 字段 | 应填内容 |
| --- | --- |
| primitives.camera_to_base_translation_m | optical frame 原点在 base 中的位置 |
| primitives.camera_to_base_xyzw | 将 optical 向量旋转到 base 的四元数 |
| primitives.tcp_offset_tool_m | 实测工具坐标中的 TCP 偏移 |
| primitives.table_z_m | 新测量的桌面高度 |
| workspace_min_m / workspace_max_m | SDK 末端有效工作区 |
| primitives.tcp_workspace_min_m / max_m | TCP 有效工作区 |
| primitives.observation_tcp_m | 实测可让开视野的位置，另做整段 IK 检查 |

项目的深度反投影来自对齐到颜色图的深度，因此要核对标定用的坐标系确实与运行时颜色内参/光学坐标一致。如果标定输出是 camera_link，需显式组合该 link 与 color optical 的静态变换，不能直接复制。

更新后保持 `commissioned=false` 和 `primitives.calibrated=false`，先恢复只读桥做核验。字段变成 true 只是启用条件，不是标定过程本身。

### 恢复项目的顺序

1. 保存本次数据和结果，关闭标定采样和标签检测节点。
2. 正常停止 ROS 相机驱动，等待 D455 释放。
3. 确认旧桥与旧静态外参节点不会造成冲突。
4. 用新配置启动 Harness 只读桥。
5. 检查新图、深度、三维定位和多个点位。
6. 运行只读原语任务，查看整段 IK 预检。
7. 现场验收后再单独安排抓取、放置与堆叠，不让标定脚本自动进入这些任务。

只读示例，分别在控制主机与已建立隧道的板端执行：

```bash
# 控制主机，确认无其他进程占用 D455
python3 -m cix_harness.baxter --config config/baxter-primitives.local.json

# 板端，VLM 与 9102 隧道已就绪；不加 --execute
python3 -m cix_harness.primitive_runner --task '抓起青色小方块' --task-kind pick
```

## 14 常见错误排查

| 现象 | 优先检查 |
| --- | --- |
| 标签完全识别不到 | 家族错误、下载错误、图像话题错误、曝光、尺寸、白边 |
| ID 正确但距离有固定比例误差 | size 是否包括了白边、打印缩放、单位、内参 |
| 姿态静止时仍跳变 | 图案太小、反光、模糊、角点质量、平面姿态歧义 |
| 标签检测有结果但没有 TF | publish_tf、standalone_tags 是否实际加载、节点命名空间 |
| 修改参数后无效 | 启动文件重新加载并覆盖、改错配置、没有重启检测节点 |
| GUI 取样失败 | 两对输入 TF 是否存在、frame 拼写、时间戳、标签是否新检测 |
| 启动时出现自动运动界面 | 是否设置 freehand_robot_movement=true；先停止自动操作 |
| RViz 出现奇怪相机位置 | dummy 外参或旧静态发布器，不把显示占位当作求解结果 |
| 误差随姿态改变 | 标签松动、夹指开合、frame 不刚性、采样同步、旋转覆盖不足 |
| 结果整体方向错误 | eye_on_hand 模式错误、相反变换、四元数顺序、optical/link 混用 |
| 相机定位正确但夹指偏移 | TCP 定义、工具长度和安装偏移 |
| 标定后 D455 被占用 | ROS 驱动未退出或旧 Harness 桥仍在运行 |
| 时效错误 | 机器人与主机时钟；桥的 offset 不会自动修复标定节点 |
| 桌面只有一处对得上 | 外参旋转、深度偏差、桌面倾斜或仅局部参考拟合 |

## 15 交付目录和记录模板

建议在控制主机保存到独立本地目录，例如 `runs/calibration/20261113-apriltag/`。不同批次使用新目录，避免覆盖。真实外参、照片和遥测不进入公开 Git。

```text
本次标定目录/
  notes.txt                 操作员、时间、硬件和版本
  tag_measurements.txt      家族、ID、横纵尺寸、采用尺寸
  mounting_photos/          相机与标签支架照片
  camera_info.yaml          实际内参与 frame
  detect_tag.launch         本次真正使用的检测启动配置
  samples/                  原始样本或导出的配对变换
  recordings/               ROS bag（若已记录）
  handeye_result.yaml        原始工具输出
  validation.txt            独立验证点、误差和结论
  config_before.json        修改前本地配置
  config_after.json         修改后本地配置
```

交接填写：

```text
操作员与日期：
源码版本 / ROS 包版本：
相机序列号 / 分辨率 / FPS：
颜色图话题 / CameraInfo 话题：
tracking_base_frame：
tracking_marker_frame：
robot_base_frame：
robot_effector_frame：
标签家族 / ID / 横向 mm / 纵向 mm / 配置 m：
标签固定方式 / 是否发生松动：
有效样本数 / 求解算法：
输出方向（必须明确）：
平移 xyz（米）：
四元数 xyzw：
独立验证姿态数 / 位置覆盖：
平移与旋转一致性结果：
独立参考点误差 / 最大误差：
左 TCP 状态 / 右 TCP 状态：
桌面测量结果：
只读定位与 IK 结果：
验收结论：通过当前范围 / 仅局部可用 / 未通过
尚未解决的问题：
配置与数据保存路径：
相机是否在采样后被移动：
```

完成后交回上述记录与数据路径，而不只是发送一张 GUI 截图。下一步真实动作由当前现场任务统一安排。

## 16 参考与阅读顺序

- [AprilTag 官方说明与尺寸定义](https://github.com/AprilRobotics/apriltag)：图案、检测尺寸和坐标系。
- [官方图案库](https://github.com/AprilRobotics/apriltag-imgs)：下载标签原图。
- [apriltag_ros](https://github.com/AprilRobotics/apriltag_ros)：检测节点和参数，以现场安装版本为准。
- [easy_handeye](https://github.com/IFL-CAMP/easy_handeye)：eye-on-base、采样与结果管理。
- [Kalibr 目标与打印说明](https://github.com/ethz-asl/kalibr/wiki/calibration-targets)：打印缩放注意；本次单标签流程不要求安装 Kalibr。
- [项目完整交接](项目完整交接.md)：项目背景、两条执行路线和已有进展。
- [原语配置模板](../config/baxter-primitives.example.json)：字段结构，不可当真实标定直接使用。

现场最短执行顺序：打印实测 → 刚性安装 → 核对 ROS 与相机 → 稳定检测 → 手动多姿态采样 → 求解 → 独立验证 → TCP 与桌面核验 → 更新本地配置 → 只读检查 → 交付记录。
