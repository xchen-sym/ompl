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
#include <ompl/base/goals/GoalState.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>
#include <ompl/control/planners/rrt/RRT.h>
#include <ompl/control/planners/sst/SST.h>
#include <ompl/base/PlannerTerminationCondition.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/config.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <Eigen/Dense>
#include <ompl/base/Goal.h>
#include <ompl/base/StateSpace.h>

namespace ob = ompl::base;
namespace oc = ompl::control;

// A GoalRegion that uses custom weights (w_pose, w_vel) only for the goal check.
class GoalWeighted : public ob::GoalRegion
{
public:
    GoalWeighted(const ob::SpaceInformationPtr &si,
                 const ob::ScopedState<ob::CompoundStateSpace> &goalState,
                 double tol, double w_x, double w_y, double w_yaw,
                 double w_v, double w_omega)
      : GoalRegion(si), goal_(goalState), tol_(tol), w_x_(w_x),
        w_y_(w_y), w_yaw_(w_yaw), w_v_(w_v), w_omega_(w_omega)
    {
        // Goal is satisfied when distanceGoal < threshold_
        setThreshold(0.0);
    }

    // Override to use our own weighted metric
    double distanceGoal(const ob::State *s) const override
    {
        // 1) unravel the compound state
        const auto *cs = s->as<ob::CompoundStateSpace::StateType>();
        auto *se2_s  = cs->as<ob::SE2StateSpace::StateType>(0);
        auto *vel_s  = cs->as<ob::RealVectorStateSpace::StateType>(1);

        const auto *csG = goal_.get();
        auto *se2_g  = csG->as<ob::SE2StateSpace::StateType>(0);
        auto *vel_g  = csG->as<ob::RealVectorStateSpace::StateType>(1);

        // 2) compute pose error
        double dx = se2_s->getX() - se2_g->getX();
        double dy = se2_s->getY() - se2_g->getY();
        double dyaw = se2_s->getYaw() - se2_g->getYaw();

        // 3) compute vel error
        double dv = vel_s->values[0] - vel_g->values[0];
        double domega = vel_s->values[1] - vel_g->values[1];

        // 4) weighted combination
        double metric = std::sqrt(w_x_ * dx * dx + w_y_ * dy * dy + w_yaw_ * dyaw * dyaw +
                                  w_v_ * dv * dv + w_omega_ * domega * domega);
        
        std::cout << "metric = " << metric << std::endl;

        return metric - tol_;
    }

private:
    ob::ScopedState<ob::CompoundStateSpace> goal_;
    double tol_, w_x_, w_y_, w_yaw_, w_v_, w_omega_;
};


struct AABB
{
    Eigen::Vector2d min, max;
    bool contains(const Eigen::Vector2d &p) const
    {
        return (p.x() >= min.x() && p.x() <= max.x()) &&
               (p.y() >= min.y() && p.y() <= max.y());
    }
};

// Your allowed boxes and bodyPoints remain unchanged:
static const std::vector<AABB> allowedBoxes = {
    {{-0.414912, -0.933520}, {1.756473, -0.459736}},
    {{-0.440026, -0.479736}, {2.079724, 0.038077}},
    {{-0.452385, 0.018077}, {2.080080, 0.507299}},
    {{-0.459443, 0.487299}, {0.452835, 0.989674}}
};

static const std::vector<Eigen::Vector2d> bodyPoints = {
    {-0.2545, 0.381},
    {-0.2545, -0.381},
    {1.16417, 0.381},
    {1.16417, -0.381}
};

bool isStateValid(const oc::SpaceInformation *si, const ob::State *state)
{
    const auto *se2 = state->as<ob::SE2StateSpace::StateType>();
    double x   = se2->getX(), y = se2->getY(), yaw = se2->getYaw();
    Eigen::Rotation2Dd R(yaw);
    for (auto &pt_body : bodyPoints)
    {
        Eigen::Vector2d w = R * pt_body + Eigen::Vector2d(x,y);
        bool insideAny = false;
        for (auto &box : allowedBoxes)
            if (box.contains(w)) { insideAny = true; break; }
        if (!insideAny)
            return false;
    }
    return si->satisfiesBounds(state);
}

void propagate(const ob::State *start, const oc::Control *control,
               const double duration, ob::State *result)
{
    // 1) Read current state and control values
    const auto *state = start->as<ob::CompoundStateSpace::StateType>();

    const auto *pos = state->as<ob::SE2StateSpace::StateType>(0);
    double x = pos->getX(), y = pos->getY(), yaw = pos->getYaw();

    const auto *vel = state->as<ob::RealVectorStateSpace::StateType>(1);
    double v = vel->values[0], omega = vel->values[1];

    const auto *acc = control->as<oc::RealVectorControlSpace::ControlType>();
    double a = acc->values[0], alpha = acc->values[1];

    // 2) Propagation
    double x_new = x + (v * duration + 0.5 * a * duration * duration) * cos(yaw);
    double y_new = y + (v * duration + 0.5 * a * duration * duration) * sin(yaw);
    double yaw_new = yaw + omega * duration + 0.5 * alpha * duration * duration;
    double v_new = v + a * duration;
    double omega_new = omega + alpha * duration;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Before Propagation:" << " x = " << x << " y = " << y << " yaw = " << yaw
              << " v = " << v << " omega = " << omega << std::endl;
    std::cout << "Propagation Acceleration:" << " a = " << a << " alpha = " << alpha << std::endl;
    std::cout << "After Propagation:" << " x = " << x_new << " y = " << y_new << " yaw = " << yaw_new
              << " v = " << v_new << " omega = " << omega_new << std::endl;
    std::cout << std::endl;

    // 3) Write into result
    auto *state_new = result->as<ob::CompoundStateSpace::StateType>();

    auto *pos_new = state_new->as<ob::SE2StateSpace::StateType>(0);
    pos_new->setX(x_new);
    pos_new->setY(y_new);
    pos_new->setYaw(yaw_new);

    auto *vel_new = state_new->as<ob::RealVectorStateSpace::StateType>(1);
    vel_new->values[0] = v_new;
    vel_new->values[1] = omega_new;
}

void plan()
{
    // 1) Set state and control spaces
    // set position space, including (x, y, yaw)
    auto pos_space(std::make_shared<ob::SE2StateSpace>());
    ob::RealVectorBounds pos_bounds(2);
    pos_bounds.setLow(0, 0.0); pos_bounds.setHigh(0, 1.0); // x position bound [0, 1]
    pos_bounds.setLow(1, 0.0); pos_bounds.setHigh(1, 1.0); // y position bound [0, 1]
    pos_space->setBounds(pos_bounds);

    // set velocity space, including (v, omega)
    auto vel_space(std::make_shared<ob::RealVectorStateSpace>(2));
    ob::RealVectorBounds vel_bounds(2);
    vel_bounds.setLow(0, 0.0); vel_bounds.setHigh(0, 4.0);  // linear velocity bound [0, 4.0]
    vel_bounds.setLow(1, -1.5); vel_bounds.setHigh(1, 1.5); // angular velocity bound [-3.0, 3.0]
    vel_space->setBounds(vel_bounds);

    // set general state space, including pos_space and vel_space
    auto space(std::make_shared<ob::CompoundStateSpace>());
    space->addSubspace(pos_space, 1.0);
    space->addSubspace(vel_space, 1.0);
    space->lock();

    // set control (acceleration) space, including (a, alpha)
    auto cspace(std::make_shared<oc::RealVectorControlSpace>(space, 2));
    ob::RealVectorBounds cbounds(2);
    cbounds.setLow(0, -1.5); cbounds.setHigh(0, 1.5); // linear acceleration bound [-1.5, 1.5]
    cbounds.setLow(1, -3.0); cbounds.setHigh(1, 3.0); // angular acceleration bound [-1.5, 1.5]
    cspace->setBounds(cbounds);

    // 2) SpaceInformation
    auto si(std::make_shared<oc::SpaceInformation>(space, cspace));
    si->setPropagationStepSize(0.01);
    // si->setMinMaxControlDuration(1, 1);
    si->setStateValidityChecker(
        [&](const ob::State *s){ return isStateValid(si.get(), s); }); // bounding box constraints
    si->setStatePropagator(propagate);

    // 3) Start & Goal
    ob::ScopedState<ob::CompoundStateSpace> start(space), goal(space);

    auto *space_start = start.get();

    auto *pos_space_start = space_start->as<ob::SE2StateSpace::StateType>(0);
    pos_space_start->setX(0.0);
    pos_space_start->setY(0.685);
    pos_space_start->setYaw(-1.570796);

    auto *vel_space_start = space_start->as<ob::RealVectorStateSpace::StateType>(1);
    vel_space_start->values[0] = 0.528379;
    vel_space_start->values[1] = 0.0;

    auto *space_goal = goal.get();

    auto *pos_space_goal = space_goal->as<ob::SE2StateSpace::StateType>(0);
    pos_space_goal->setX(0.86329);
    pos_space_goal->setY(0.0);
    pos_space_goal->setYaw(0.0);

    auto *vel_space_goal = space_goal->as<ob::RealVectorStateSpace::StateType>(1);
    vel_space_goal->values[0] = 0.930091;
    vel_space_goal->values[1] = 0.0;

    // 4) ProblemDefinition + Optimization
    auto pdef(std::make_shared<ob::ProblemDefinition>(si));
    // pdef->setStartAndGoalStates(start, goal, 0.05);

    pdef->addStartState(start);
    // customize each state's weight while setting convergence target
    double tol = 0.1, w_x = 1.0, w_y = 1.0, w_yaw = 1.0, w_v = 1.0, w_omega = 1.0;
    auto weighted_goal = std::make_shared<GoalWeighted>(si, goal, tol, w_x, w_y,
                                                        w_yaw, w_v, w_omega);
    pdef->setGoal(weighted_goal);

    auto opt = std::make_shared<ob::PathLengthOptimizationObjective>(si);
    pdef->setOptimizationObjective(opt);

    // 5) Use SST for near-optimal kinodynamic planning
    // auto planner(std::make_shared<oc::RRT>(si));
    auto planner = std::make_shared<oc::SST>(si);
    planner->setProblemDefinition(pdef);
    planner->setup();

    // 6) Solve
    si->printSettings(std::cout);
    pdef->print(std::cout);

    ob::PlannerStatus status;
    const double increment = 0.002;
    const double max_plan_time = 2.0;
    double plan_time = 0.0;
    // Planning incrementally until solver finds a solution or
    // maximum allowed time is exceeded
    while (status != ob::PlannerStatus::StatusType::EXACT_SOLUTION)
    {
        auto ptc = ob::timedPlannerTerminationCondition(increment);
        status = planner->solve(ptc);
        plan_time += increment;
        if (plan_time > max_plan_time) break;
    }

    auto path = pdef->getSolutionPath();
    if (status == ob::PlannerStatus::StatusType::EXACT_SOLUTION)
    {
        std::cout << "Found solution:" << std::endl;
        path->print(std::cout);
        std::cout << "Planning took " << plan_time << " seconds\n";
        std::cout << "Planner status: " << status << std::endl;
    }
    else
    {
        std::cout << "No solution found\n";
        if (status == ob::PlannerStatus::StatusType::APPROXIMATE_SOLUTION)
        {
            std::cout << "Found approximate solution:" << std::endl;
            path->print(std::cout);
        }
        std::cout << "Planner status: " << status << std::endl;
    }
}

int main()
{
    std::cout << "OMPL version: " << OMPL_VERSION << std::endl;
    plan();
    return 0;
}
