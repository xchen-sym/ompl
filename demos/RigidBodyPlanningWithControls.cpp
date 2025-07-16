/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2010, Rice University
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Rice University nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/* Author: Ioan Sucan */

#include <ompl/control/SpaceInformation.h>
#include <ompl/base/Goal.h>
#include <ompl/base/StateSpace.h>
#include <ompl/base/goals/GoalState.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>
#include <ompl/control/planners/rrt/RRT.h>
#include <ompl/control/planners/sst/SST.h>
#include <ompl/base/PlannerTerminationCondition.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/config.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <math.h>

namespace ob = ompl::base;
namespace oc = ompl::control;

struct Waypoint {
    double x, y, yaw, v, omega;
};

struct BoundingBox {
    double minX, maxX, minY, maxY;
};

void propagate(const ob::State *start, const oc::Control *control,
               const double duration, ob::State *result)
{
    // 1) Read current state and control values
    const auto *pos = start->as<ob::SE2StateSpace::StateType>();
    double x = pos->getX(), y = pos->getY(), yaw = pos->getYaw();

    const auto *vel = control->as<oc::RealVectorControlSpace::ControlType>();
    double v = vel->values[0], omega = vel->values[1];

    // 2) Propagation
    double x_new = x + v * duration * cos(yaw);
    double y_new = y + v * duration * sin(yaw);
    double yaw_new = yaw + omega * duration;

    // std::cout << std::fixed << std::setprecision(4);
    // std::cout << "Before Propagation:" << " x = " << x << " y = " << y 
    //           << " yaw = " << yaw << std::endl;
    // std::cout << "Propagation Acceleration:" << " v = " << v << " omega = " << omega << std::endl;
    // std::cout << "After Propagation:" << " x = " << x_new << " y = " << y_new 
    //           << " yaw = " << yaw_new << std::endl;
    // std::cout << std::endl;

    // 3) Write into result
    auto *pos_new = result->as<ob::SE2StateSpace::StateType>();
    pos_new->setX(x_new);
    pos_new->setY(y_new);
    pos_new->setYaw(yaw_new);
}

class GoalWeighted : public ob::GoalRegion
{
public:
    GoalWeighted(const ob::SpaceInformationPtr &si,
                 const ob::ScopedState<ob::SE2StateSpace> &goalState,
                 double tol, double w_x, double w_y, double w_yaw)
      : GoalRegion(si), goal_(goalState), tol_(tol),
        w_x_(w_x), w_y_(w_y), w_yaw_(w_yaw)
    {
        // Goal is satisfied when distanceGoal < threshold_
        setThreshold(0.0);
    }

    // Override to use our own weighted metric
    double distanceGoal(const ob::State *s) const override
    {
        // 1) unravel the current and goal state
        const auto *current_state = s->as<ob::SE2StateSpace::StateType>();
        const auto *goal_state = goal_.get();

        // 2) compute state error
        double dx = current_state->getX() - goal_state->getX();
        double dy = current_state->getY() - goal_state->getY();
        double dyaw = current_state->getYaw() - goal_state->getYaw();

        // 4) weighted combination
        double metric = std::sqrt(w_x_ * dx * dx + w_y_ * dy * dy + w_yaw_ * dyaw * dyaw);
        
        // std::cout << "metric = " << metric << std::endl;

        return metric - tol_;
    }

private:
    ob::ScopedState<ob::SE2StateSpace> goal_;
    double tol_, w_x_, w_y_, w_yaw_;
};

class TrajectoryGenerator
{
private:
    Waypoint start_, goal_;
    std::vector<BoundingBox> allowedBoxes_;
    const std::vector<Eigen::Vector2d> bodyPoints_;

    BoundingBox posBounds_;
    // velocity and acceleration constraints
    const double v_max_, omega_max_, a_max_, alpha_max_;

    const double ctrl_dt_; // control cycle time
    const double tol_, w_x_, w_y_, w_yaw_; // goal convergence tolerance and weights
    const double plan_dt_, plan_time_; // planning incremental and total time
    const double v_ref_; // reference velocity when generating real trajectory

    std::vector<Eigen::Vector3d> pathProfiles_; // path points generated by OMPL

    std::vector<Waypoint> trajectoryPoints_; // trajectory points
    int current_path_idx_; // Track current position along the path

    oc::SpaceInformationPtr si_;
    ob::ProblemDefinitionPtr pdef_;
    std::shared_ptr<oc::SST> planner_;

    void setStateAndControlSpace();
    void setPropagation();
    void setBoundingBoxConstraints();
    void setStartAndGoal();
    void setPlanner();

    bool stateValidityCheck(const ob::State *state);

public:
    TrajectoryGenerator(const Waypoint &start,
                        const Waypoint &goal,
                        const std::vector<BoundingBox> &allowed,
                        const std::vector<Eigen::Vector2d> &bodyPts,
                        const BoundingBox &posBounds,
                        const double v_max, const double omega_max,
                        const double a_max, const double alpha_max,
                        const double ctrl_dt, const double tol,
                        const double w_x, const double w_y,
                        const double w_yaw, const double plan_dt,
                        const double plan_time, const double v_ref);

    ob::PlannerStatus plan();

    void KinematicSimulation();

    const std::vector<Eigen::Vector3d>& getPathProfiles() const;

    const std::vector<Waypoint>& getTrajectoryPoints() const;

    const std::pair<Waypoint, Waypoint> getInitialAndFinalWaypoints() const;
};

TrajectoryGenerator::TrajectoryGenerator(const Waypoint &start,
                                         const Waypoint &goal,
                                         const std::vector<BoundingBox> &allowed,
                                         const std::vector<Eigen::Vector2d> &bodyPts,
                                         const BoundingBox &posBounds,
                                         const double v_max, const double omega_max,
                                         const double a_max, const double alpha_max,
                                         const double ctrl_dt, const double tol,
                                         const double w_x, const double w_y,
                                         const double w_yaw, const double plan_dt,
                                         const double plan_time, const double v_ref)
  : start_(start), goal_(goal), allowedBoxes_(allowed), bodyPoints_(bodyPts),
    posBounds_(posBounds), v_max_(v_max), omega_max_(omega_max), a_max_(a_max),
    alpha_max_(alpha_max), ctrl_dt_(ctrl_dt), tol_(tol), w_x_(w_x), w_y_(w_y),
    w_yaw_(w_yaw), plan_dt_(plan_dt), plan_time_(plan_time), v_ref_(v_ref),
    current_path_idx_(0)
{
    setStateAndControlSpace();
    setPropagation();
    setBoundingBoxConstraints();
    setStartAndGoal();
    setPlanner();
};

void TrajectoryGenerator::setStateAndControlSpace()
{
    // 1) set state space, including [x, y, yaw]
    auto sspace(std::make_shared<ob::SE2StateSpace>());
    ob::RealVectorBounds sbounds(2);
    sbounds.setLow(0, posBounds_.minX); sbounds.setHigh(0, posBounds_.maxX); // x position bound
    sbounds.setLow(1, posBounds_.minY); sbounds.setHigh(1, posBounds_.maxY); // y position bound
    sspace->setBounds(sbounds);

    // 2) set control (velocity) space, including [v, omega]
    auto cspace(std::make_shared<oc::RealVectorControlSpace>(sspace, 2));
    ob::RealVectorBounds cbounds(2);
    cbounds.setLow(0, 0.0); cbounds.setHigh(0, v_max_); // linear velocity v bound
    cbounds.setLow(1, -omega_max_); cbounds.setHigh(1, omega_max_); // angular velocity omega bound
    cspace->setBounds(cbounds);

    // 3) set SpaceInformation + ProblemDefinition
    si_ = std::make_shared<oc::SpaceInformation>(sspace, cspace);
    pdef_ = std::make_shared<ob::ProblemDefinition>(si_);
};

void TrajectoryGenerator::setPropagation()
{
    si_->setPropagationStepSize(ctrl_dt_);
    si_->setMinMaxControlDuration(1, 10);
    si_->setStatePropagator(propagate);
};

bool TrajectoryGenerator::stateValidityCheck(const ob::State *state)
{
    const auto *se2 = state->as<ob::SE2StateSpace::StateType>();
    double x = se2->getX(), y = se2->getY(), yaw = se2->getYaw();
    Eigen::Rotation2Dd R(yaw);
    for (auto &pt_body : bodyPoints_)
    {
        Eigen::Vector2d w = R * pt_body + Eigen::Vector2d(x,y);
        bool insideAny = false;
        for (auto &box : allowedBoxes_)
            if (w.x() >= box.minX && w.x() <= box.maxX &&
                w.y() >= box.minY && w.y() <= box.maxY)
            { insideAny = true; break;}
        if (!insideAny)
            return false;
    }
    return si_->satisfiesBounds(state);
};

void TrajectoryGenerator::setBoundingBoxConstraints()
{
    si_->setStateValidityChecker(
        [&](const ob::State *state){ return stateValidityCheck(state); });
};

void TrajectoryGenerator::setStartAndGoal()
{
    ob::ScopedState<ob::SE2StateSpace> start(si_->getStateSpace()),
                                       goal(si_->getStateSpace());

    start->setX(start_.x);
    start->setY(start_.y);
    start->setYaw(start_.yaw);

    goal->setX(goal_.x);
    goal->setY(goal_.y);
    goal->setYaw(goal_.yaw);

    pdef_->addStartState(start);
    auto w_goal = std::make_shared<GoalWeighted>(
        si_, goal, tol_, w_x_, w_y_, w_yaw_);
    pdef_->setGoal(w_goal);

    auto opt = std::make_shared<ob::PathLengthOptimizationObjective>(si_);
    pdef_->setOptimizationObjective(opt);
};

void TrajectoryGenerator::setPlanner()
{
    planner_ = std::make_shared<oc::SST>(si_);
    planner_->setProblemDefinition(pdef_);
    planner_->setup();
};

ob::PlannerStatus TrajectoryGenerator::plan()
{
    ob::PlannerStatus status;
    double t = 0.0;

    // incremental planning until solution is found or max time is exceeded
    while (status != ob::PlannerStatus::EXACT_SOLUTION)
    {
        auto ptc = ob::timedPlannerTerminationCondition(plan_dt_);
        status = planner_->solve(ptc);
        t += plan_dt_;
        if (t > plan_time_) break;
    }

    auto path = pdef_->getSolutionPath();
    if (status == ob::PlannerStatus::EXACT_SOLUTION)
    {
        // store path profiles
        auto ctrlPath = std::dynamic_pointer_cast<oc::PathControl>(path);
        pathProfiles_.clear();
        // 1) store the first path point
        const auto *s0 = ctrlPath->getState(0)->as<ob::SE2StateSpace::StateType>();
        pathProfiles_.emplace_back(s0->getX(), s0->getY(), s0->getYaw());
        // 2) replay every control step
        for (std::size_t i = 0; i < ctrlPath->getControlCount(); ++i)
        {
            const auto* control = ctrlPath->getControl(i);
            unsigned int numSteps = std::round(ctrlPath->getControlDuration(i) / ctrl_dt_);

            ob::State *prev = si_->allocState();
            si_->copyState(prev, ctrlPath->getState(i));

            for (unsigned int s = 0; s < numSteps; ++s)
            {
                ob::State *next = si_->allocState();
                propagate(prev, control, ctrl_dt_, next);
                const auto *st = next->as<ob::SE2StateSpace::StateType>();
                pathProfiles_.emplace_back(st->getX(), st->getY(), st->getYaw());
                si_->freeState(prev);
                prev = next;
            }
            si_->freeState(prev);
        }

        std::cout << "Found solution:" << std::endl;
        path->print(std::cout);
        std::cout << "Planning took " << t << " seconds\n";
        std::cout << "Planner status: " << status << std::endl;
    }
    else
    {
        std::cout << "No solution found\n";
        if (status == ob::PlannerStatus::APPROXIMATE_SOLUTION)
        {
            std::cout << "Found approximate solution:" << std::endl;
            path->print(std::cout);
        }
        std::cout << "Planner status: " << status << std::endl;
    }

    return status;
};

void TrajectoryGenerator::KinematicSimulation()
{
    // store start waypoint
    trajectoryPoints_.clear();
    trajectoryPoints_.emplace_back(start_);
    Waypoint wp = start_;
    current_path_idx_ = 0;  // Reset path index
    const double lookahead_dist = 0.1;  // Lookahead distance for path following

    // Track minimum error within tol_
    double min_error = std::numeric_limits<double>::max();
    bool found_valid_solution = false;

    // Real trajectory time length should not be too long. Here it is set within
    // 2 * reference trajectory length.
    for (std::size_t step = 0; step < 2 * pathProfiles_.size(); ++step)
    {
        // Find closest path point index and update current_path_idx_
        int closest_idx = 0;
        double min_dist = std::numeric_limits<double>::max();
        for (std::size_t i = 0; i < pathProfiles_.size(); ++i)
        {
            double dx = pathProfiles_[i][0] - wp.x;
            double dy = pathProfiles_[i][1] - wp.y;
            double dist = std::hypot(dx, dy);
            if (dist < min_dist)
            {
                min_dist = dist;
                closest_idx = i;
            }
        }
        current_path_idx_ = std::max(current_path_idx_, closest_idx);

        // Find the target point using lookahead
        int target_idx = current_path_idx_;
        double cumulative_dist = 0.0;
        
        // Look ahead along the path until we reach the desired lookahead distance
        for (std::size_t i = current_path_idx_; i < pathProfiles_.size() - 1; ++i)
        {
            double dx = pathProfiles_[i + 1][0] - pathProfiles_[i][0];
            double dy = pathProfiles_[i + 1][1] - pathProfiles_[i][1];
            double seg_dist = std::hypot(dx, dy);
            
            if (cumulative_dist + seg_dist >= lookahead_dist)
            {
                target_idx = i + 1;
                break;
            }
            cumulative_dist += seg_dist;
            target_idx = i + 1;
        }
        
        // Ensure we don't go backwards
        target_idx = std::max(target_idx, current_path_idx_);

        // velocity control - use adaptive speed based on path curvature
        double v_ref = v_ref_;
        // Target goal waypoint velocity if the bot is so close to the goal
        // that it will reach this target velocity under acceleration constraints.
        double dist_to_goal = std::hypot(wp.x - goal_.x, wp.y - goal_.y);
        double dist_to_goal_threshold =
            std::abs(goal_.v - v_ref_) * (goal_.v + v_ref_) / (2.0 * a_max_);
        if (dist_to_goal < dist_to_goal_threshold)
        {
            v_ref = goal_.v;
        }
        
        double dv = std::clamp(v_ref - wp.v, -a_max_ * ctrl_dt_, a_max_ * ctrl_dt_);
        wp.v = std::clamp(wp.v + dv, 0.0, v_max_);
        
        double omega_tgt = 0.0;
        double kp_angular = 10.0;  // Proportional gain for yaw control
        
        // When close to goal, blend between path following and goal alignment
        const double goal_approach_dist = 0.2;  // Distance threshold for goal approach behavior
        if (dist_to_goal < goal_approach_dist)
        {
            // Calculate blend factor based on distance to goal (0 = far, 1 = at goal)
            double blend_factor = 1.0 - (dist_to_goal / goal_approach_dist);
            blend_factor = std::clamp(blend_factor, 0.0, 1.0);
            
            // Calculate angular error for path following
            double yaw_error = goal_.yaw - wp.yaw;
            while (yaw_error > M_PI) yaw_error -= 2.0 * M_PI;
            while (yaw_error < -M_PI) yaw_error += 2.0 * M_PI;
            
            // Calculate target angular velocity for path following
            double omega_pos = kp_angular * yaw_error;
            
            // Blend between path-following omega and goal omega
            omega_tgt = (1.0 - blend_factor) * omega_pos + blend_factor * goal_.omega;
        }
        else
        {
            // Normal path following behavior, bot should head towards reference point.
            double target_x = pathProfiles_[target_idx][0];
            double target_y = pathProfiles_[target_idx][1];
            double yaw_tgt_pos = std::atan2(target_y - wp.y, target_x - wp.x);
        
            double yaw_error = yaw_tgt_pos - wp.yaw;
            while (yaw_error > M_PI) yaw_error -= 2.0 * M_PI;
            while (yaw_error < -M_PI) yaw_error += 2.0 * M_PI;

            // Calculate target angular velocity for path following
            omega_tgt = kp_angular * yaw_error;
        }
        
        // Apply angular acceleration constraints
        double domega = std::clamp(
            omega_tgt - wp.omega, -alpha_max_ * ctrl_dt_, alpha_max_ * ctrl_dt_);
        wp.omega = std::clamp(wp.omega + domega, -omega_max_, omega_max_);

        // status update
        wp.x += wp.v * std::cos(wp.yaw) * ctrl_dt_;
        wp.y += wp.v * std::sin(wp.yaw) * ctrl_dt_;
        wp.yaw += wp.omega * ctrl_dt_;
        
        // normalize yaw angle
        while (wp.yaw > M_PI) wp.yaw -= 2.0 * M_PI;
        while (wp.yaw < -M_PI) wp.yaw += 2.0 * M_PI;

        // Check if this is the best waypoint so far
        double pos_error = std::hypot(wp.x - goal_.x, wp.y - goal_.y);
        double yaw_error = wp.yaw - goal_.yaw;
        while (yaw_error > M_PI) yaw_error -= 2.0 * M_PI;
        while (yaw_error < -M_PI) yaw_error += 2.0 * M_PI; 
        double total_error = std::hypot(pos_error, yaw_error);
        
        // Track minimum error that's within tolerance
        if (total_error < tol_ && total_error < min_error)
        {
            min_error = total_error;
            found_valid_solution = true;
        }
        
        // Early stopping: Break if we have a valid solution and error is increasing
        if (found_valid_solution && total_error > min_error) break;

        // store trajectory points
        trajectoryPoints_.emplace_back(wp);
    }
}

const std::vector<Eigen::Vector3d>& TrajectoryGenerator::getPathProfiles() const
{
    return pathProfiles_;
};

const std::vector<Waypoint>& TrajectoryGenerator::getTrajectoryPoints() const
{
    return trajectoryPoints_;
};

const std::pair<Waypoint, Waypoint> TrajectoryGenerator::getInitialAndFinalWaypoints() const
{
    return std::make_pair(start_, goal_);
}

int main()
{
    const Waypoint start{0.0, 0.685, -1.570796, 0.528379, 0.0};
    const Waypoint goal{0.86329, 0.0, 0.0, 0.930091, 0.0};
    const std::vector<BoundingBox> allowed = {
        {-0.414912, 1.756473, -0.933520, -0.459736},
        {-0.440026, 2.079724, -0.479736, 0.038077},
        {-0.452385, 2.080080, 0.018077, 0.507299},
        {-0.459443, 0.452835, 0.487299, 0.989674}
    };
    const std::vector<Eigen::Vector2d> bodyPts = {
        {-0.2545, 0.381},
        {-0.2545, -0.381},
        {1.16417, 0.381},
        {1.16417, -0.381}
    };
    const BoundingBox posBounds = {0.0, 1.0, 0.0, 1.0};
    const double v_max = 2.0, omega_max = 1.5, a_max = 1.5, alpha_max = 15.0,
                 ctrl_dt = 0.01, tol = 0.05, w_x = 1.0, w_y = 1.0, w_yaw = 1.0,
                 plan_dt = 0.002, plan_time = 2.0, v_ref = 0.5;

    TrajectoryGenerator tg(start, goal, allowed, bodyPts, posBounds, v_max,
        omega_max, a_max, alpha_max, ctrl_dt, tol, w_x, w_y, w_yaw, plan_dt,
        plan_time, v_ref);

    ob::PlannerStatus status = tg.plan();
    if (status == ob::PlannerStatus::EXACT_SOLUTION)
    {
        // Print initial and final waypoints
        const auto waypoints = tg.getInitialAndFinalWaypoints();
        std::cout << "Initial Waypoint: [" << waypoints.first.x << ", " << waypoints.first.y 
                  << ", " << waypoints.first.yaw << ", " << waypoints.first.v 
                  << ", " << waypoints.first.omega << "]" << std::endl;
        std::cout << "Final Waypoint: [" << waypoints.second.x << ", " << waypoints.second.y 
                  << ", " << waypoints.second.yaw << ", " << waypoints.second.v 
                  << ", " << waypoints.second.omega << "]" << std::endl;
        std::cout << std::endl;

        const auto &path = tg.getPathProfiles();
        std::cout << "Path:" << std::endl;
        for (const auto &p : path)
            std::cout << "[" << p.x() << ", " << p.y() << ", " << p.z() << "]" << std::endl;
        
        // generate trajectory profile
        tg.KinematicSimulation();
        const auto &traj = tg.getTrajectoryPoints();
        std::cout << "Trajectory:" << std::endl;
        for (const auto &t : traj)
            std::cout << "[" << t.x << ", " << t.y << ", " << t.yaw << ", " << t.v << ", " << t.omega << "]" << std::endl;

        std::cout << "Planning successful." << std::endl;
    }
    else
    {
        std::cout << "Planning failed." << std::endl;
    }
    return 0;
};