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
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <Eigen/Dense>

namespace ob = ompl::base;
namespace oc = ompl::control;

// A GoalRegion that uses custom weights (w_pose, w_vel) only for the goal check.
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
        
        std::cout << "metric = " << metric << std::endl;

        return metric - tol_;
    }

private:
    ob::ScopedState<ob::SE2StateSpace> goal_;
    double tol_, w_x_, w_y_, w_yaw_;
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
    const auto *pos = start->as<ob::SE2StateSpace::StateType>();
    double x = pos->getX(), y = pos->getY(), yaw = pos->getYaw();

    const auto *vel = control->as<oc::RealVectorControlSpace::ControlType>();
    double v = vel->values[0], omega = vel->values[1];

    // 2) Propagation
    double x_new = x + v * duration * cos(yaw);
    double y_new = y + v * duration * sin(yaw);
    double yaw_new = yaw + omega * duration;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Before Propagation:" << " x = " << x << " y = " << y 
              << " yaw = " << yaw << std::endl;
    std::cout << "Propagation Acceleration:" << " v = " << v << " omega = " << omega << std::endl;
    std::cout << "After Propagation:" << " x = " << x_new << " y = " << y_new 
              << " yaw = " << yaw_new << std::endl;
    std::cout << std::endl;

    // 3) Write into result
    auto *pos_new = result->as<ob::SE2StateSpace::StateType>();
    pos_new->setX(x_new);
    pos_new->setY(y_new);
    pos_new->setYaw(yaw_new);
}

void plan()
{
    // 1) Set state and control spaces
    // set state space, including (x, y, yaw)
    auto space(std::make_shared<ob::SE2StateSpace>());
    ob::RealVectorBounds bounds(2);
    bounds.setLow(0, 0.0); bounds.setHigh(0, 1.0); // x position bound [0, 1]
    bounds.setLow(1, 0.0); bounds.setHigh(1, 1.0); // y position bound [0, 1]
    space->setBounds(bounds);

    // set control (velocity) space, including (v, omega)
    auto cspace(std::make_shared<oc::RealVectorControlSpace>(space, 2));
    ob::RealVectorBounds cbounds(2);
    cbounds.setLow(0, 0.0); cbounds.setHigh(0, 2.0); // linear velocity bound [0.0, 2.0]
    cbounds.setLow(1, -1.5); cbounds.setHigh(1, 1.5); // angular velocity bound [-1.5, 1.5]
    cspace->setBounds(cbounds);

    // 2) SpaceInformation
    auto si(std::make_shared<oc::SpaceInformation>(space, cspace));
    si->setPropagationStepSize(0.01);
    // si->setMinMaxControlDuration(1, 1);
    si->setStateValidityChecker(
        [&](const ob::State *s){ return isStateValid(si.get(), s); }); // bounding box constraints
    si->setStatePropagator(propagate);

    // 3) Start & Goal
    ob::ScopedState<ob::SE2StateSpace> start(space), goal(space);

    start->setX(0.0);
    start->setY(0.685);
    start->setYaw(-1.570796);

    goal->setX(0.86329);
    goal->setY(0.0);
    goal->setYaw(0.0);

    // 4) ProblemDefinition + Optimization
    auto pdef(std::make_shared<ob::ProblemDefinition>(si));
    // pdef->setStartAndGoalStates(start, goal, 0.05);

    pdef->addStartState(start);
    // customize each state's weight while setting convergence target
    double tol = 0.05, w_x = 1.0, w_y = 1.0, w_yaw = 1.0;
    auto weighted_goal = std::make_shared<GoalWeighted>(
        si, goal, tol, w_x, w_y, w_yaw);
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
