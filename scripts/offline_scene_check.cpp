// In-memory MoveIt 1 state validity for an archived Baxter PlanningScene.
// One stdin line per configuration (left seven joints then right, SDK order).
// No ROS master, publisher, subscriber or controller is constructed.
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <moveit/planning_scene/planning_scene.h>
#include <ros/serialization.h>
#include <srdfdom/model.h>
#include <urdf_parser/urdf_parser.h>

std::string read_file(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("Cannot open " + path);
  return std::string(std::istreambuf_iterator<char>(stream), {});
}

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  ros::Time::init();
  auto urdf = urdf::parseURDF(read_file(argv[1]));
  auto srdf = std::make_shared<srdf::Model>();
  if (!urdf || !srdf->initString(*urdf, read_file(argv[2]))) return 3;
  auto model = std::make_shared<moveit::core::RobotModel>(urdf, srdf);
  planning_scene::PlanningScene scene(model);
  auto bytes = read_file(argv[3]);
  if (bytes.empty()) return 4;
  ros::serialization::IStream stream(reinterpret_cast<uint8_t*>(&bytes[0]), bytes.size());
  moveit_msgs::PlanningScene snapshot;
  ros::serialization::deserialize(stream, snapshot);
  if (snapshot.is_diff || snapshot.robot_state.is_diff || snapshot.robot_model_name != model->getName()) return 5;
  scene.setPlanningSceneMsg(snapshot);
  auto& state = scene.getCurrentStateNonConst();
  for (const auto& body : snapshot.robot_state.attached_collision_objects) {
    if (!state.hasAttachedBody(body.object.id)) return 6;
  }
  for (const auto& object : snapshot.world.collision_objects) {
    if (!scene.getWorld()->hasObject(object.id)) return 6;
  }
  std::vector<std::string> names;
  for (const auto& arm : {"left", "right"}) {
    for (const auto& joint : {"s0", "s1", "e0", "e1", "w0", "w1", "w2"}) {
      names.push_back(std::string(arm) + "_" + joint);
    }
  }
  collision_detection::CollisionRequest request;
  request.contacts = true;
  request.max_contacts = 4;
  for (std::string line; std::getline(std::cin, line);) {
    std::istringstream fields(line);
    for (const auto& name : names) {
      double value;
      if (!(fields >> value) || !std::isfinite(value)) return 7;
      state.setVariablePosition(name, value);
    }
    std::string extra;
    if (fields >> extra) return 7;
    state.update();
    collision_detection::CollisionResult result;
    scene.checkCollision(request, result, state);
    std::cout << "BAXTER_FCL ";
    if (result.collision) {
      std::cout << "collision";
      for (const auto& contact : result.contacts) {
        std::cout << " " << contact.first.first << "/" << contact.first.second;
      }
    } else if (!state.satisfiesBounds(1e-6)) {
      std::cout << "joint_bounds";
    } else {
      std::cout << "clear";
    }
    std::cout << '\n';
  }
}
