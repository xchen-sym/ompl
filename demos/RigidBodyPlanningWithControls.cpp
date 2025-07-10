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

    std::vector<Eigen::Vector3d> pathProfiles_; // path points generated by OMPL

    std::vector<Waypoint> trajectoryPoints_; // trajectory points

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
                        const double plan_time);

    ob::PlannerStatus plan();

    void KinematicSimulation();

    const std::vector<Eigen::Vector3d>& getPathProfiles() const;

    const std::vector<Waypoint>& getTrajectoryPoints() const;
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
                                         const double plan_time)
  : start_(start), goal_(goal), allowedBoxes_(allowed), bodyPoints_(bodyPts),
    posBounds_(posBounds), v_max_(v_max), omega_max_(omega_max), a_max_(a_max),
    alpha_max_(alpha_max), ctrl_dt_(ctrl_dt), tol_(tol), w_x_(w_x), w_y_(w_y),
    w_yaw_(w_yaw), plan_dt_(plan_dt), plan_time_(plan_time)
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

    for (int step = 0; step < 200; ++step)
    {
        // find nearest path point
        int target_idx = 0;
        double min_dist = std::numeric_limits<double>::max();
        for (std::size_t i = 0; i < pathProfiles_.size(); ++i)
        {
            double dx = pathProfiles_[i][0] - wp.x;
            double dy = pathProfiles_[i][1] - wp.y;
            double dist = std::hypot(dx, dy);
            if (dist < min_dist)
            {
                min_dist = dist;
                target_idx = i;
            }
        }

        // velocity control
        double target_v = v_max_;
        double dv = std::clamp(target_v - wp.v, -a_max_ * ctrl_dt_, a_max_ * ctrl_dt_);
        wp.v = std::clamp(wp.v + dv, 0.0, v_max_);
        
        // yaw angle (steering) control
        // double omega_min = std::max(-omega_max_, wp.omega - alpha_max_ * ctrl_dt_); // minimum angular velocity available
        // double omega_max = std::min(omega_max_, wp.omega + alpha_max_ * ctrl_dt_); // maximum angular velocity available

        double target_yaw = pathProfiles_[target_idx][2]; // target yaw angle
        double target_omega = (target_yaw - wp.yaw) / ctrl_dt_; // target angular velocity
        while (target_omega > M_PI) target_omega -= 2.0 * M_PI;
        while (target_omega < -M_PI) target_omega += 2.0 * M_PI;
        // wp.omega = std::clamp(target_omega, omega_min, omega_max);
        wp.omega = std::clamp(target_omega, -omega_max_, omega_max_);

        // status update
        wp.x += wp.v * std::cos(wp.yaw) * ctrl_dt_;
        wp.y += wp.v * std::sin(wp.yaw) * ctrl_dt_;
        wp.yaw += wp.omega * ctrl_dt_;
        while (wp.yaw > M_PI) wp.yaw -= 2.0 * M_PI;
        while (wp.yaw < -M_PI) wp.yaw += 2.0 * M_PI;

        // store trajectory points
        trajectoryPoints_.emplace_back(wp);

        // early stop when goal region is reached
        if (std::hypot(std::hypot(wp.x - goal_.x, wp.y - goal_.y), wp.yaw - goal_.yaw) < tol_)
            break;
    }
}

// void TrajectoryGenerator::KinematicSimulation()
// {
//     Waypoint wp = start_;
//     double L = 1.0;

//     trajectoryPoints_.clear();
//     trajectoryPoints_.emplace_back(wp);

//     for (int step = 0; step < 200; ++step)
//     {
//         // find closest path point
//         int target_idx = 0;
//         double min_dist = std::numeric_limits<double>::max();
//         for (std::size_t i = 0; i < pathProfiles_.size(); ++i)
//         {
//             double dx = pathProfiles_[i][0] - wp.x;
//             double dy = pathProfiles_[i][1] - wp.y;
//             double dist = std::hypot(dx, dy);
//             if (dist < min_dist)
//             {
//                 min_dist = dist;
//                 target_idx = i;
//             }
//         }

//         const auto &target = pathProfiles_[target_idx];
//         double tx = target[0];
//         double ty = target[1];
//         double tyaw = target[2];

//         // yaw control (P-control)
//         double alpha = tyaw - wp.yaw;
//         while (alpha > M_PI) alpha -= 2.0 * M_PI;
//         while (alpha < -M_PI) alpha += 2.0 * M_PI;

//         double steer = std::clamp(alpha, -M_PI / 4.0, M_PI / 4.0);

//         // velocity control (P-control)
//         double target_v = v_max_;
//         double dv = std::clamp(target_v - wp.v, -a_max_ * ctrl_dt_, a_max_ * ctrl_dt_);
//         wp.v += dv;
//         wp.v = std::clamp(wp.v, 0.0, v_max_);

//         // update status
//         wp.x += wp.v * std::cos(wp.yaw) * ctrl_dt_;
//         wp.y += wp.v * std::sin(wp.yaw) * ctrl_dt_;
//         wp.omega = (wp.v / L) * std::tan(steer);
//         wp.omega = std::clamp(wp.omega, -omega_max_, omega_max_);
//         wp.yaw += wp.omega * ctrl_dt_;

//         // normalize yaw angle
//         while (wp.yaw > M_PI) wp.yaw -= 2.0 * M_PI;
//         while (wp.yaw < -M_PI) wp.yaw += 2.0 * M_PI;

//         // store new trajectory point
//         trajectoryPoints_.emplace_back(wp);

//         // early stop if goal region is reached
//         if (std::hypot(wp.x - goal_.x, wp.y - goal_.y) < tol_)
//             break;
//     }
// }

const std::vector<Eigen::Vector3d>& TrajectoryGenerator::getPathProfiles() const
{
    return pathProfiles_;
};

const std::vector<Waypoint>& TrajectoryGenerator::getTrajectoryPoints() const
{
    return trajectoryPoints_;
};

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
    const double v_max = 2.0, omega_max = 1.5, a_max = 1.5, alpha_max = 1.5,
                 ctrl_dt = 0.01, tol = 0.05, w_x = 1.0, w_y = 1.0,
                 w_yaw = 1.0, plan_dt = 0.002, plan_time = 2.0;

    TrajectoryGenerator tg(start, goal, allowed, bodyPts, posBounds,
                           v_max, omega_max, a_max, alpha_max, ctrl_dt,
                           tol, w_x, w_y, w_yaw, plan_dt, plan_time);

    ob::PlannerStatus status = tg.plan();
    if (status == ob::PlannerStatus::EXACT_SOLUTION)
    {
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