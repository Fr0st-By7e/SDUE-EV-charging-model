# Bi-Criterion SDUE EV Charging Model
This repository contains a simplified Python implementation of a bi-criterion Stochastic Dynamic User Equilibrium (SDUE) model for analyzing electric vehicle (EV) charging behavior in urban transportation networks. Includes MSA-based equilibrium, flexibility analysis, and case studies comparing fixed vs load-responsive pricing in urban networks.

---

## Abstract
This report presents the implementation and analysis of a simplified bi-criterion Stochastic Dynamic User Equilibrium (SDUE) model for electric vehicle (EV) charging integrated with trip chain behaviour. The study focuses on modelling EV driver decisions considering both travel time and charging cost within a time-dependent transportation network. The implemented model incorporates fast and slow charging strategies, multinomial logit-based charging station selection, and dynamic pricing mechanisms. A Method of Successive Averages (MSA) approach is used to achieve convergence in network conditions and pricing.
Due to the absence of real-world datasets, a synthetic network and EV population are generated to emulate realistic behaviour. Two scenarios are analysed: one without pricing incentives and another with dynamic pricing to regulate charging demand. Results demonstrate the impact of pricing strategies on load distribution, charging behaviour, and overall system efficiency. The model successfully captures key interactions between transportation and energy systems, offering insights into demand-side management for EV charging infrastructure.

---

## Introduction

### Background
The global transition toward sustainable transportation has led to a rapid increase in the adoption of electric vehicles (EVs). Unlike conventional internal combustion engine vehicles, EVs rely on battery storage, which introduces new operational constraints such as limited driving range, charging time, and dependence on charging infrastructure. These constraints significantly influence travel behaviour and route choice decisions.
Traditional transportation models primarily focus on minimizing travel time or distance, assuming unlimited fuel availability. However, such models are inadequate for EV systems where drivers must also consider battery state-of-charge (SOC), availability of charging stations, and associated costs. Simultaneously, from a power systems perspective, uncoordinated EV charging can lead to severe peak loads, grid instability, and inefficient energy utilization.

### Problem Statement
The key challenge addressed in this study is the integration of EV charging behaviour into traffic assignment models under realistic conditions. Specifically, the problem involves:
-	Modelling EV driver decisions under energy constraints
-	Capturing the interaction between transportation networks and charging infrastructure
-	Incorporating both travel time and monetary cost into decision-making
-	Ensuring system-level equilibrium under stochastic user behaviour

### Importance & Application
This integrated modelling framework has several practical applications:
-	Urban Planning: Helps in optimal placement of charging stations
-	Smart Grid Management: Enables demand response and peak load reduction
-	Policy Design: Assists in evaluating pricing incentives and regulations
-	Transportation Optimization: Improves travel efficiency and reduces congestion
By addressing these aspects, the study contributes to the development of intelligent and sustainable EV ecosystems.

---

## Summary

### Core Idea
The research paper introduces a bi-criterion Stochastic Dynamic User Equilibrium (SDUE) model that simultaneously considers travel time and charging cost when modelling EV driver behaviour. Unlike deterministic models, this framework accounts for variability in user preferences and decision-making through probabilistic choice models.

### Key Contributions
1.	Integration of Trip Chains
Instead of treating trips independently, the model considers sequences of trips (e.g., Home–Work–Home), allowing for more realistic representation of daily travel patterns.
2.	Bi-Criterion Utility Function
The utility function incorporates:
-	Travel time (representing convenience and delay)
-	Charging cost (representing economic factors)
3.	Charging Decision Modelling
The model differentiates between:
-	Fast Charging (FCS): High power, shorter duration
-	Slow Charging (SCS): Lower power, typically at destinations
4.	Multinomial Logit (MNL) Model
Used to represent stochastic choice behaviour when selecting charging stations.
5.	Dynamic Pricing Mechanism
Charging prices are adjusted based on demand, encouraging load balancing across time and space.
6.	Equilibrium Framework (SDUE)
The system reaches equilibrium when no user can improve their utility by unilaterally changing their decision.

### Assumptions
-	Users act rationally but with probabilistic variation
-	Charging stations are available based on predefined probabilities
-	Traffic congestion affects travel time
-	Energy consumption depends on speed

---

## Methodology

### Overall Framework
The methodology follows a simulation-based iterative framework that integrates transportation dynamics, energy consumption, and user decision-making.

#### Step 1: Network and Population Generation
-	A directed graph represents the transportation network
-	Nodes represent zones and charging locations
-	Links represent roads with attributes like distance and free-flow travel time
-	EV population is generated with:
1.	Home locations
2.	Trip chains (H-W-H or H-O-H)
3.	Initial battery SOC

#### Step 2: Time Discretization
-	The day is divided into fixed intervals (15 minutes)
-	Traffic conditions and charging loads are evaluated per time bin

#### Step 3: Travel & Energy Modelling
For each trip:
-	Shortest path is computed using time-dependent link costs
-	Travel time is calculated
-	Energy consumption is estimated based on:
1.	Distance
2.	Speed-dependent efficiency function

#### Step 4: Charging Decision Logic
##### Fast Charging Decision
An EV opts for fast charging if:
-	Remaining battery after trip falls below a minimum threshold
-	Future trip requirements cannot be satisfied
##### Slow Charging Decision
Occurs at destinations based on:
-	Parking duration
-	Availability of charging infrastructure
-	Future energy requirements

#### Step 5: Charging Station Selection
-	All feasible charging stations are evaluated
-	Utility for each option is computed:
1.	Travel time + charging time
2.	Charging cost
-	A logit model determines selection probability

#### Step 6: Utility Calculation
The utility function is defined as: <br>
U = -(\alpha_{tra} \cdot t + \alpha_{mon} \cdot c)  <br>
[LaTeX representation] <br>
This ensures that:
-	Higher time or cost reduces utility
-	Users prefer lower-cost and faster options

#### Step 7: Traffic & Load Update (MSA)
The Method of Successive Averages (MSA) is used to update:
-	Link travel times (based on congestion)
-	Charging prices (based on load levels)
<br>
Update rule: <br>
X_{new} = \frac{1}{k+1} X_{current} + \frac{k}{k+1} X_{previous} <br>
[LaTeX representation] <br>
This ensures smooth convergence.

#### Step 8: Convergence Check
-	Average utility across EVs is monitored
-	If change falls below a threshold → convergence achieved

### Implementation Specifications
-	Synthetic network instead of real-world data
-	Simplified congestion model
-	No queuing delays at charging stations
-	Approximate energy consumption function

---

## Implementation Details

### Architecture
The implementation is modular and object-oriented:
-	Data Classes → EV, Trip, Link
-	Network Module → shortest path & travel time
-	Energy Model → consumption estimation
-	Charging Logic → decision rules
-	Simulation Engine → iterative equilibrium computation
-	Analysis Module → visualization and export

### Key Components
-	TimeDependentNetwork → manages travel times
-	PricingModel → handles dynamic pricing
-	EVChargingSDUEModel → core simulation logic

### Libraries Used
-	NumPy → numerical computations
-	NetworkX → graph algorithms
-	Matplotlib → visualization

---

## Results & Inference

### Case 1 and Case 2 Fast Charging Load Profiles
<img width="600" height="300" alt="case1_no_price_incentives_fast_load_profiles" src="https://github.com/user-attachments/assets/6eb8e520-1e2d-46e9-9b51-007bf695c78a" />
<img width="600" height="300" alt="case2_price_incentives_fast_load_profiles" src="https://github.com/user-attachments/assets/4abef579-25e4-4acc-9463-17665762b5fd" />
These plots illustrate how charging demand varies throughout the day. Case 1 shows sharp peaks with high average due to unregulated charging, whereas Case 2 demonstrates smoother load distribution due to pricing incentives.

***

### Case 1 and Case 2 Slow Charging Summary
<img width="400" height="250" alt="case1_no_price_incentives_slow_summary" src="https://github.com/user-attachments/assets/941809f9-f882-4764-812c-0e232bb19bd8" />
<img width="400" height="250" alt="case2_price_incentives_slow_summary" src="https://github.com/user-attachments/assets/6cf1ef06-61fe-4353-9d59-98ec4d891eea" />
Slow charging contributes significantly to flexibility. Increased deferrable energy in Case 2 indicates slightly better utilization of off-peak charging opportunities as sample size is relatively small.

***

### Case 1 and Case 2 Average Trip Chain Utility
<img width="400" height="250" alt="case1_no_price_incentives_convergence" src="https://github.com/user-attachments/assets/06745450-ad98-4d08-97f1-a87a9754a09e" />
<img width="400" height="250" alt="case2_price_incentives_convergence" src="https://github.com/user-attachments/assets/93216e83-ee58-4d7c-aa3a-8a1905dcf8b9" />
The convergence pattern shows stability of the system. Slight utility reduction in Case 2 is expected due to added cost, but overall efficiency improves. The first few iterations show deviant behaviour as we consider zero traffic congestion and hence requires certain no. of iterations to be considered valid.

***

### Case Comparison of Total Fast Charging Loads
<img width="500" height="250" alt="case_comparison_total_fast_load" src="https://github.com/user-attachments/assets/7b30fd44-e38d-4eea-b51e-c07096e8e286" />
Dynamic pricing effectively reduces peak loads and spreads demand more evenly across time.

***

### Case 2 FCS Charging Prices
<img width="600" height="300" alt="case2_price_incentives_prices" src="https://github.com/user-attachments/assets/c17d848d-5b50-4345-b0ea-bcefbd735a12" />
Dynamic pricing changes with increase in demand at a particular station as demonstrated by the graph.

---

## Conclusion

### Key Takeaways
-	Integrated EV charging and traffic modelling is essential
-	Dynamic pricing significantly improves system efficiency
-	Trip-chain modelling enhances behavioural accuracy

### Limitations
- Lack of real-world data
-	Simplified charging and congestion models
-	No queue modelling at stations

### Future Scope
-	Advanced charging station modelling
-	Integration with real datasets

---

## Appendix

#### Reference: [Flexibility potential of electric vehicle charging: A trip chain analysis under bi-criterion stochastic dynamic user equilibrium - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666792425000344?via%3Dihub)
