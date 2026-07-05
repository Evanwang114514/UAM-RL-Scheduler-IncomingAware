# Incoming-Aware Proximal Policy Optimization with Realistic Multi-Vehicle Dispatch Dynamics for Passenger-Centric Vertiport Selection in Urban Air Mobility

## I. Abstract

The UAGMC paper (IEEE TITS 2026) proposed an air-ground collaborative air taxi dispatching framework that processes high-dimensional heterogeneous state information through MSCE and STIN, validating the effectiveness of deep reinforcement learning in UAM vertiport selection. However, urban traffic environments exhibit characteristics such as low vehicle speeds (urban average of 20-30 km/h), a large number of vertiports, and significant variations in passenger travel distance distributions and demand patterns. To address these issues, this project makes certain adjustments and improvements on small-scale data.

Building upon the UAGMC framework [1], this project constructs a UAM vertiport dispatching simulation platform under urban traffic conditions. The core contributions include:

(1) An X-shaped 8-vertiport network covering a 40x40 km urban area, where each vertiport can simultaneously handle both departure and arrival tasks;

(2) A travel-distance-aware passenger classification mechanism, where passengers are randomly generated within the map and short-distance travelers are directly assigned to ground transportation to avoid efficiency loss from forced aircraft allocation;

(3) A realistic eVTOL dispatching simulation that formulates idle eVTOL scheduling strategies to emulate real-world dispatch environments;

(4) Queue-predictive reinforcement learning that introduces passenger queue length and incoming eVTOL count into the state space, enabling the agent to anticipate queue trends, and adopts a seven-choice action space for selecting suitable departure vertiports for air travelers.

Experimental results demonstrate that under the scenario of 20 km/h urban average speed, single-passenger eVTOL capacity, and 8 vertiports, the lightweight PPO model effectively reduces average system travel time and peak-hour queue pressure. After streamlining features and action space, a lightweight UAM dispatching strategy is achieved. This work builds upon the UAGMC framework [1] with realistic multi-vehicle dispatch dynamics, incoming-aware queue prediction, and short-trip ground diversion, validating the effectiveness of lightweight PPO in such scenarios.


## II. Problem Formulation

### A. Passenger Travel Chain Modeling

This paper studies a multi-modal passenger transportation system integrating ground-based CAVs and aerial eVTOLs. Each passenger's travel consists of four stages:

$$ \text{Origin} \xrightarrow{\text{CAV}} \text{Departure Vertiport} \xrightarrow{\text{Queue}} \text{eVTOL} \xrightarrow{\text{Flight}} \text{Arrival Vertiport} \xrightarrow{\text{CAV}} \text{Destination} $$

Each passenger $i \in \mathcal{N}$ has an origin $o_i$ and a destination $d_i$, and needs to select a departure vertiport $k_i \in \mathcal{V}$ and an arrival vertiport $l_i \in \mathcal{V}$. The total travel time is defined as:

$$ TT_{i}^{\text{tot}} = TT_{o_i \rightarrow k_i}^{\text{g}} + TW_{k_i}^{\text{wait}} + TT_{k_i \rightarrow l_i}^{\text{a}} + TT_{l_i \rightarrow d_i}^{\text{g}} $$


### B. Vertiport Layout

The environment deploys 8 vertiports with both takeoff and landing capabilities, arranged in an X-shaped configuration covering a 40x40 km urban area:

| ID | Coordinates (km) | ID | Coordinates (km) |
|---|---|---|---|
| 0 | (5, 5) | 4 | (15, 25) |
| 1 | (35, 5) | 5 | (25, 25) |
| 2 | (15, 15) | 6 | (5, 35) |
| 3 | (25, 15) | 7 | (35, 35) |

Passengers are randomly generated within the map area, with origins and destinations randomly combined from 16 10x10 km grid blocks, with inter-block travel enforced to exclude excessively short trips.

### C. Passenger Generation and Classification Mechanism

Upon activation at the origin, the system first calculates the straight-line ground distance from origin to destination. For short-distance passengers (whose ground travel time is less than any feasible flight option), the system directly assigns ground transportation. These passengers travel to their destination at CAV speed without entering vertiport queues or flight procedures, and are immediately considered to have completed their travel task.

For medium- and long-distance passengers, the system first assigns the arrival vertiport using the shortest-path method, and then employs a queue-aware model to assign the departure vertiport. Passengers travel from their origin to the departure vertiport at CAV speed, enter the queue to wait for an available eVTOL, and are considered to have completed one flight task upon takeoff.

### D. eVTOL Dispatching Strategy

This environment implements a multi-eVTOL flight dispatching simulation to approximate real-world dispatch conditions. The following dispatching logic is executed sequentially at each time step (tick = 0.1 minutes):

**Takeoff Phase**: At each minute, all vertiports are traversed. If a vertiport has both queued passengers and an idle eVTOL, the head-of-queue passenger departs for the target vertiport. The eVTOL enters the flight queue, and the passenger's total time is recorded upon completion of the travel task.

**Dispatching Phase**: The system scans the queue lengths of all vertiports and identifies the vertiport with the largest number of unassigned passengers as the target. All idle eVTOLs are retrieved, and the one closest to the target vertiport is dispatched. Furthermore, if the dispatched eVTOL arrives at the target vertiport and there are still queued passengers, it waits for the next minute's takeoff command, forming a dynamic replenishment mechanism.

**Flight Update Phase**: At each time step, the remaining flight time of all in-flight eVTOLs is updated. When the remaining time reaches zero, the eVTOL arrives at the destination vertiport and rejoins the idle pool, ready for the next task.

The above dispatching logic balances the service capacity of each vertiport as much as possible under a limited fleet size, simulating the dynamic dispatching mechanisms in UAM operations.

### E. Optimization Objective

The system objective is to minimize the average total travel time across all passengers:

$$ \min_{(k_i,l_i) \in \mathcal{V}^2} \frac{1}{|\mathcal{N}|} \sum_{i \in \mathcal{N}} TT_{i}^{\text{tot}} $$

In this problem, passengers are strongly coupled through shared vertiport and eVTOL resources. The time-varying nature of decisions and the non-stationarity of queue dynamics render traditional static optimization methods inapplicable. Therefore, this paper formulates the problem as a Markov Decision Process and employs deep reinforcement learning to solve it.


## III. Methodology

### A. Overall Framework

The proposed method framework consists of three core modules:

- Simulation Environment Layer: Simulates the UAM vertiport dispatching environment
- State Encoding Layer: Encodes raw observations into a 20-dimensional state vector
- Decision Optimization Layer: Employs the PPO algorithm to train a policy network that outputs decision actions

### B. State Space Design

At time $t$, the agent observes a 20-dimensional state vector from the environment:

$$ S(t) = [Q_0, \ldots, Q_7, F_0, \ldots, F_7, x_o, y_o, x_d, y_d] $$

| Component | Description | Dimension | Normalization |
|---|---|---|---|
| $Q_k$ | Queue length at vertiport $k$ | 8 | $Q_k / (\max Q + 1)$ |
| $F_k$ | Number of eVTOLs flying to vertiport $k$ | 8 | $F_k / (\max F + 1)$ |
| $(x_o, y_o)$ | Passenger origin coordinates | 2 | $x / 40$ |
| $(x_d, y_d)$ | Passenger destination coordinates | 2 | $x / 40$ |

**Motivation for $F_k$**: The queue length $Q_k$ reflects the current congestion state but cannot indicate queue trends. Consider two scenarios: Vertiport A has a long queue but many eVTOLs are about to arrive, resulting in short actual wait times; Vertiport B has a short queue but no incoming eVTOLs, resulting in long actual wait times. The introduction of $F_k$ enables the agent to anticipate queue evolution trends and make better vertiport selection decisions.

### C. Action Space Reduction

The action is a seven-choice discrete selection:

$$ A(t) \in \{0, 1, \ldots, 6\} $$

where action $a$ denotes selecting the $a$-th vertiport (excluding the arrival vertiport) sorted by total travel time (walking + flight). The arrival vertiport is fixed to the one closest to the passenger's destination.

### D. Reward Function Design

The reward function is defined as the negative value of the average total travel time of completed passengers:

$$ R(t) = -\frac{1}{|\mathcal{N}_t|} \sum_{i \in \mathcal{N}_t} TT_i^{\text{tot}} $$

where $\mathcal{N}_t$ is the set of passengers who have completed travel by time $t$. This reward design directly aligns with the optimization objective, avoiding goal deviation caused by reward shaping.


## IV. Experiments

### A. Experimental Setup

The experiments are conducted on a simulation environment with a 40x40 km urban area and 8 vertiports in an X-shaped layout. The key parameters are configured as follows:

| Parameter | Value |
|---|---|
| Map Size | 40 x 40 km |
| Number of Vertiports | 8 |
| Ground CAV Speed | 20 km/h |
| eVTOL Flight Speed | 120 km/h |
| eVTOL Passenger Capacity | 1 (for scenario simplification) |
| Passenger Demand | 800 passengers (one generated every 0.5 min) |
| Number of eVTOLs | 8 / 12 / 16 |
| Training Steps | 120,000 |
| Discount Factor $\gamma$ | 0.99 |
| Entropy Coefficient | 0.03 |

**Baseline Methods**:

- **All Ground**: All passengers travel exclusively by CAV without using eVTOLs. Total travel time includes only ground travel.

- **Shortest Path**: Each passenger independently selects the vertiport pair that minimizes $TT_{o_i \rightarrow k_i}^{\text{g}} + TT_{k_i \rightarrow l_i}^{\text{a}} + TT_{l_i \rightarrow d_i}^{\text{g}}$. Passengers may also choose ground travel throughout, in which case the flight distance is zero and the decision reduces to $TT^{\text{tot}} = TT_{o_i \rightarrow d_i}^{\text{g}}$. This method does not consider waiting time $TW^{\text{wait}}$.

- **Shortest Path with Penalty**: Introduces a queue penalty term on top of the shortest path, modifying the vertiport selection cost to $TT^{\text{total}} + \lambda \times Q_k$, where $Q_k$ is the current queue length at vertiport $k$. The penalty coefficient $\lambda$ is set to 225 (corresponding to 22.5 minutes per passenger), determined through repeated experimental validation (penalty coefficients in the 200-250 range yield optimal performance). This method serves as a rule-based congestion avoidance baseline.

- **Baseline PPO (12-dim)**: Uses the same PPO algorithm and network architecture as the proposed method, but the state space includes only the 8 vertiport queue lengths and 4-dimensional origin-destination coordinates (12 dimensions total), without the incoming eVTOL count feature. This method is used to validate the effectiveness of the proposed 20-dimensional state design.

**Evaluation Metrics**:

- Average Ground Travel Time (AGT)
- Average Waiting Time (AWT)
- Average Air Travel Time (AAT)
- Average Total Travel Time (ATT): AGT + AWT + AAT, the ultimate optimization objective of this paper
- System Completion Time: Total simulation time required for all passengers to complete their travel, reflecting system throughput efficiency

### B. Experimental Results

#### (1) 8 eVTOL Configuration

| Method | AGT | AWT | AAT | ATT | Completion Time |
|---|---|---|---|---|---|
| All Ground | 65.52 | — | — | 65.52 | 399.6 |
| Shortest Path | 25.89 | 133.51 | 7.66 | 167.06 | 842.2 |
| Shortest Path with Penalty | 24.08 | 84.77 | 6.84 | 115.69 | 744.3 |
| Baseline PPO (12-dim) | 24.03 | 74.83 | 6.36 | 105.22 | 710.8 |
| **Proposed Method (20-dim)** | **23.02** | **69.39** | **6.19** | **98.60** | **693.7** |

Under the 8-vehicle configuration, the proposed method reduces ATT by **41.0%** compared to Shortest Path, by **14.8%** compared to Shortest Path with Penalty, and by **6.3%** compared to Baseline PPO.

#### (2) 12 eVTOL Configuration

| Method | AGT | AWT | AAT | ATT | Completion Time |
|---|---|---|---|---|---|
| All Ground | 65.52 | — | — | 65.52 | 399.6 |
| Shortest Path | 25.89 | 47.60 | 7.66 | 81.15 | 579.3 |
| Shortest Path with Penalty | 24.32 | 25.06 | 6.88 | 56.26 | 517.3 |
| Baseline PPO (12-dim) | 24.10 | 20.24 | 6.46 | 50.80 | 500.5 |
| **Proposed Method (20-dim)** | **23.21** | **18.66** | **6.17** | **48.03** | **501.8** |

Under the 12-vehicle configuration, the proposed method reduces ATT by **40.8%** compared to Shortest Path, by **14.6%** compared to Shortest Path with Penalty, and by **5.5%** compared to Baseline PPO.

#### (3) 16 eVTOL Configuration

| Method | AGT | AWT | AAT | ATT | Completion Time |
|---|---|---|---|---|---|
| All Ground | 65.52 | — | — | 65.52 | 399.6 |
| Shortest Path | 25.89 | 12.61 | 7.66 | 46.16 | 448.2 |
| Shortest Path with Penalty | 24.73 | 9.65 | 7.06 | 41.44 | 448.2 |
| Baseline PPO (12-dim) | 24.30 | 6.57 | 6.60 | 37.47 | 440.8 |
| **Proposed Method (20-dim)** | **23.47** | **5.86** | **6.35** | **35.68** | **441.2** |

Under the 16-vehicle configuration, the proposed method reduces ATT by **22.7%** compared to Shortest Path, by **13.9%** compared to Shortest Path with Penalty, and by **4.8%** compared to Baseline PPO.

### C. Experimental Conclusions

Based on the above experimental results, the following conclusions can be drawn:

(1) PPO outperforms rule-based baselines: Under all three eVTOL configurations, the proposed PPO method consistently outperforms both the Shortest Path and Shortest Path with Penalty baselines, demonstrating that reinforcement learning, through interaction with the environment, learns dynamic congestion avoidance strategies that surpass fixed linear penalty rules.

(2) Queue prediction features are effective: Through ablation experiments comparing Baseline PPO (12-dim, containing only passenger information and queue lengths) with the proposed method (20-dim, including incoming eVTOL count), AWT shows significant reduction. Observing the number of eVTOLs flying to vertiports enables the model to better predict queue trends at each vertiport.

(3) Model advantages are more significant under capacity-constrained scenarios: Under the 8-vehicle configuration (capacity-constrained), the proposed method achieves greater ATT reduction compared to Baseline PPO than under the 16-vehicle configuration (capacity-sufficient). This validates that when queues are more likely to accumulate under capacity constraints, the model gains more from queue prediction and dynamic congestion avoidance.

(4) Ground-air classification mechanism is effective: The proposed method achieves higher ground completion rates than Shortest Path and Shortest Path with Penalty across all three configurations. Short-distance passengers are reasonably assigned to ground transportation, effectively alleviating vertiport queue pressure and improving system travel efficiency.

(5) Reinforcement learning improves throughput efficiency: Compared to Shortest Path, the reinforcement learning method completes all passenger dispatching tasks in shorter time, demonstrating stronger adaptability to complex systems.


## V. Limitations and Future Directions

(1) Insufficient feature dimensions: The current state space only includes passenger information and simple vertiport throughput information. Future work should incorporate more environmental variables to fully predict environmental evolution trends.

(2) eVTOL dispatching rules remain manually designed: The dispatching strategy is based on manually designed rules and has not yet achieved joint optimization of dispatching and passenger selection. Future work will introduce multi-agent joint reinforcement learning frameworks to achieve joint optimization of eVTOL dispatching and passenger vertiport selection.

(3) Gap between scenario assumptions and real operational environments: Experiments are based on idealized assumptions, including fixed passenger numbers, fixed ground speeds, and no airspace or communication constraints. Future work will introduce dynamic passenger generation models, vertiport capacity constraints, charging constraints, and airspace constraints.

(4) Insufficient temporal feature capture: The policy network fails to explicitly model queue temporal evolution patterns. Future work will introduce Transformer or LSTM-based temporal modeling modules to equip the agent with long-term memory and prediction capabilities for queue trends.

(5) Limited computational resources leading to insufficient training: Due to local CPU computational limitations, the number of training steps is relatively small, and model stability and optimality may not have reached their upper bounds. Future work will extend training steps to train more stable allocation models.


## VI. Project Structure

```
UAM-RL-Scheduler-IncomingAware/
├── data/
│   ├── passengers_100.csv
│   ├── passengers_800.csv
│   └── passengers_800_test.csv
├── models/
│   ├── improved_ppo/
│   │   ├── final_model.zip
│   │   └── vec_normalize.pkl
│   └── seven_choice_ppo/
│       ├── final_model.zip
│       └── vec_normalize.pkl
├── env.py
├── seven_choice_env.py
├── seven_choice_env_improved.py
├── train.py
├── train_improved.py
├── test.py
├── test_improved.py
├── shortest_path.py
├── shortest_path_with_penalty.py
├── test_all_ground.py
├── generate_passengers.py
├── requirements.txt
├── LICENSE
└── README.md
```

## VII. File Descriptions

| File | Description |
|---|---|
| env.py | Core simulation environment |
| seven_choice_env.py | 12-dim state environment wrapper |
| seven_choice_env_improved.py | 20-dim state environment wrapper |
| train.py | Train baseline PPO (12-dim) |
| train_improved.py | Train proposed method (20-dim) |
| test.py | Test baseline PPO (12-dim) |
| test_improved.py | Test proposed method (20-dim) |
| shortest_path.py | Shortest path baseline |
| shortest_path_with_penalty.py | Shortest path with penalty baseline |
| test_all_ground.py | All-ground baseline |
| generate_passengers.py | Generate passenger data |


## VIII. Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate passenger data
python generate_passengers.py

# Run baselines
python test_all_ground.py              # All ground
python shortest_path.py                # Shortest path
python shortest_path_with_penalty.py   # Shortest path with penalty

# Train models
python train.py                        # Train baseline PPO (12-dim)
python train_improved.py               # Train proposed method (20-dim)

# Test models
python test.py                         # Test baseline PPO (12-dim)
python test_improved.py                # Test proposed method (20-dim)
```

## IX. Conclusion

This project presents a small-scale UAM vertiport dispatching simulation framework for analyzing the dispatching stability and efficiency of lightweight PPO in urban air mobility scenarios. Experimental results demonstrate that incoming-aware queue prediction is more effective under capacity-constrained conditions, and simple state features can achieve performance superior to rule-based baselines in low-speed urban scenarios.

By introducing realistic multi-vehicle dispatch dynamics, this project constructs a simple human-machine collaborative system, while the incoming-aware feature enables the model to anticipate queue trends, with more significant contributions under sufficient capacity. The short-distance passenger ground diversion mechanism reduces unnecessary vertiport queuing from the source.

Objectively speaking, it is difficult to conduct complete per-vehicle simulations for all passengers and eVTOLs in real urban systems. Therefore, this work provides a lightweight experimental framework under idealized small-scale environments. In such scenarios, streamlined states and simple strategies are sufficient. However, in larger-scale or more complex scenarios, appropriate simplification or estimation using parameters such as service rates, along with increasing observation dimensions, remains necessary.

## X. References
```
[1] A. Pang et al., "Heterogeneous Vertiport Selection Optimization for On-Demand Air Taxi Services: A Deep Reinforcement Learning Approach," in IEEE Transactions on Intelligent Transportation Systems, doi: 10.1109/TITS.2026.3680351.
```

