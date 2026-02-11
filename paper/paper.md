---
title: 'OTEX: an open source Python package for ocean thermal energy conversion modeling'
tags:
  - Python
  - ocean energy
  - OTEC
  - thermodynamics
  - renewable energy
  - techno-economic analysis
authors:
  - name: Manuel Soto Calvo
    orcid: 0000-0003-4312-6300
    affiliation: 1
  - name: Han Soo Lee
    orcid: 0000-0001-7749-0317
    affiliation: "1, 2, 3"
  - name: Zachary Williams
    orcid: 0009-0007-4964-1393
    affiliation: 1
affiliations:
 - name: Coastal Hazards and Energy System Science Laboratory, Hiroshima University, Japan
   index: 1
 - name: Center for Planetary Health and Innovation Science (PHIS), The IDEC institute, Hiroshima University
   index: 2
 - name: Smart Energy, Graduate School of Innovation and Practice for Smart Society, Hiroshima University
   index: 3
date: 11 February 2026
bibliography: paper.bib
---

# Summary

Ocean Thermal Energy Conversion (OTEC) exploits the temperature gradient between warm surface seawater and cold deep ocean water to generate continuous, renewable baseload power. Despite a theoretical potential estimated at 300 exajoules per year [@IRENA2014], OTEC commercialization has been hindered by high capital costs and technical complexities [@Langer2020; @Xiao2023].

**OTEX** is an open-source Python-based framework designed to address the scarcity of comprehensive modeling tools in this sector. It provides a unified environment for the design, simulation, and technoeconomic evaluation of OTEC systems [@SotoCalvo2025; @Nakib2025]. The software capabilities include:

* **Thermodynamic Cycle Modeling:** Implementation of Closed, Open, and Hybrid Rankine cycles, as well as advanced Kalina and Uehara cycles using pure and mixed working fluids (e.g., Ammonia, R134a, Propane, Ammonia-Water mixtures) [@Samsuri2021].
* **Resource Integration:** Automatic retrieval and processing of global oceanographic data via the Copernicus Marine Environment Monitoring Service (CMEMS) for site-specific analysis [@CopernicusMarine2024].
* **Technoeconomic Analysis:** Calculation of Levelized Cost of Electricity (LCOE) based on component scaling laws, coupled with uncertainty quantification via Monte Carlo simulations and Sobol sensitivity analysis [@Langer2020; @Ghaedi2024].

# Statement of need

The transition to a net-zero energy landscape requires reliable baseload renewables to complement intermittent sources like wind and solar [@IEA2021; @Notton2018]. OTEC offers capacity factors exceeding 90% [@Langer2024; @Brecha2021], yet its advancement is constrained by a lack of accessible software tools.

Existing analytical frameworks for OTEC are often proprietary, limited to specific cycle configurations, or implemented in platforms that restrict accessibility for researchers in resource-constrained settings [@Xiao2023; @Nakib2025]. This "software gap" prevents systematic exploration of the design space and rigorous feasibility studies.

`OTEX` was developed to fill this gap. It allows researchers, engineers, and students to:

1.  **Simulate complex thermodynamics:** Handling the non-linear properties of zeotropic mixtures in Kalina and Uehara cycles [@Gonidaki2026; @Brodal2019].
2.  **Analyze Off-Design Performance:** Modeling how plants behave when seasonal water temperatures deviate from the nominal design point using sliding pressure control strategies [@Langer2024].
3.  **Quantify Risk:** Using built-in statistical tools to assess economic viability under uncertainty [@Shields2016; @Azzini2021].

The software is written in Python 3.9+ and leverages the scientific Python ecosystem (`NumPy` [@Harris2020], `SciPy` [@Virtanen2020], `Pandas` [@McKinney2010]) and utilizes `CoolProp` [@Bell2014] for high-fidelity fluid property calculations.

# State of the field

While general process simulators (e.g., Aspen Plus, DWSIM) exist, they are general-purpose tools that lack specific OTEC modules for oceanographic data integration, cold water pipe hydraulics, and specialized marine heat exchanger sizing.

Previous specific OTEC models in the literature have typically been ad-hoc scripts not released as maintained software packages [@Vera2020; @Hall2022]. `OTEX` distinguishes itself by being a fully modular, documented, and extensible package available via PyPI. It uniquely integrates the thermodynamic core with:

* **Hydraulic modeling:** Explicit calculation of pumping losses (Darcy-Weisbach) and pipe sizing [@Kowalczuk2020].
* **Geo-spatial capabilities:** Enabling regional mapping of LCOE and power output, as demonstrated in the included case study of Cuban waters [@SotoCalvo2025].

# Mathematics

`OTEX` solves coupled systems of non-linear equations to determine thermodynamic state points across multiple cycle architectures.

## Closed Rankine Cycle

For the baseline Closed Rankine cycle, the evaporation pressure $P_{evap}$ is determined by the Minimum Internal Temperature Approach (MITA) constraint at the evaporator pinch point:

$$T_{sat}(P_{evap}) = T_{WW,out} + MITA_{evap}$$

The warm seawater outlet temperature $T_{WW,out}$ is determined simultaneously from the evaporator energy balance:

$$\dot{m}_{WW} c_{p,sw} (T_{WW,in} - T_{WW,out}) = \dot{m}_{wf} (h_2 - h_1)$$

These equations are solved iteratively via the secant method [@Pal2023]. The condensation pressure follows an analogous formulation:

$$T_{sat}(P_{cond}) = T_{CW,out} - MITA_{cond}$$

The turbine expansion (2→3) is modeled with isentropic efficiency $\eta_t$:

$$h_3 = h_2 - \eta_t (h_2 - h_{3s})$$

where $h_{3s}$ satisfies the isentropic condition $s_{3s} = s_2$ at pressure $P_{cond}$. Gross turbine power output is:

$$\dot{W}_t = \dot{m}_{wf} (h_2 - h_3)$$

The feed pump work for incompressible liquid compression is:

$$\dot{W}_p = \frac{\dot{m}_{wf} v_f (P_{evap} - P_{cond})}{\eta_p}$$

The cycle thermal efficiency is defined as:

$$\eta_{th} = \frac{\dot{W}_t - \dot{W}_p}{\dot{m}_{wf}(h_2 - h_1)}$$

For OTEC applications with $\Delta T \approx 20°C$, this typically ranges from 2.5% to 4.0% [@Bekiloglu2025].

## Open and Hybrid Rankine Cycles

In the Open (Claude) cycle, warm seawater itself is the working fluid via flash evaporation at sub-atmospheric pressure [@Rajagopalan2013]:

$$P_{flash} = P_{sat}(T_{WW,in} - \Delta T_{NEL})$$

where $\Delta T_{NEL}$ represents non-equilibrium thermal losses. The vaporized mass fraction is:

$$x_{flash} = \frac{c_p (T_{WW,in} - \Delta T_{NEL} - T_{sat}(P_{flash}))}{h_{fg}}$$

The Hybrid cycle combines a closed ammonia cycle with flash steam, yielding total power [@Wu2014; @Yue2018]:

$$\dot{W}_{hybrid} = \dot{W}_{closed} + \dot{W}_{flash}$$

## Kalina and Uehara Cycles

The Kalina cycle employs ammonia-water (NH$_3$-H$_2$O) binary mixtures, exploiting variable boiling temperature to reduce exergy destruction [@Gonidaki2026]. Mixture enthalpy at temperature $T$, pressure $P$, and ammonia concentration $x$ is:

$$h_{mix}(T, P, x) = x \cdot h_{NH_3}(T,P) + (1-x) \cdot h_{H_2O}(T,P) + h_{ex}(T, P, x)$$

Phase equilibrium in the separator is governed by equality of component fugacities [@Lemmon2018; @Bell2014]:

$$y_{NH_3} \phi_{NH_3}^V P = x_{NH_3} \gamma_{NH_3} f_{NH_3}^{0,L}$$

$$y_{H_2O} \phi_{H_2O}^V P = x_{H_2O} \gamma_{H_2O} f_{H_2O}^{0,L}$$

The Uehara cycle extends the Kalina cycle with absorption heat pump principles [@Gao2025; @Matsuda2018]. The absorber energy balance couples mass and heat transfer:

$$\dot{Q}_{abs} = \dot{m}_{lean}(h_{lean,in} - h_{lean,out}) + \dot{m}_{vapor}(h_{vapor} - h_{condensate})$$

with mass conservation:

$$\dot{m}_{basic} x_{basic} = \dot{m}_{rich} x_{rich} + \dot{m}_{lean} x_{lean}$$

## Off-Design Performance

OTEX implements sliding pressure control for off-design operation. Under warm-water deficit ($T_{WW} < T_{WW,design}$), the evaporation pressure is reduced [@Langer2024]:

$$P_{evap,off} = P_{sat}(T_{WW,actual} - \Delta T_{WW} - MITA_{evap})$$

Heat exchanger off-design performance is modeled via the NTU-effectiveness method [@Qiao2025]:

$$NTU = \frac{U \cdot A}{C_{min}}$$

where the heat transfer coefficient $U$ varies with flow regime:

$$U \propto Re^n$$

with exponent $n \approx 0.8$ for turbulent flow. Annual energy production is computed across monthly time steps using CMEMS temperature data [@CopernicusMarine2024]:

$$E_{annual} = \sum_{i=1}^{12} \dot{W}_{net,i} \cdot \Delta t_i$$

The capacity factor is defined as [@Langer2024; @VanZwieten2017]:

$$CF = \frac{E_{annual}}{\dot{W}_{gross,design} \times 8760}$$

## Seawater System Hydraulics

Parasitic power consumption in seawater pumping constitutes 30--40% of gross power. The total pump power accounts for static head and frictional losses [@Kowalczuk2020]:

$$\dot{W}_{pump} = \frac{\dot{m}_{sw}}{\rho_{sw}} \cdot \frac{\rho_{sw} g H + \Delta P_f}{\eta_{pump}}$$

where frictional losses are calculated via the Darcy-Weisbach equation:

$$\Delta P_f = f \frac{L}{D} \frac{\rho_{sw} v^2}{2}$$

The net power output is then:

$$\dot{W}_{net} = \dot{W}_t - \dot{W}_{p,wf} - \dot{W}_{pump,CW} - \dot{W}_{pump,WW} - \dot{W}_{aux}$$

## Economic Modeling

The Levelized Cost of Electricity (LCOE) uses a discounted cash flow formulation [@Langer2020]:

$$LCOE = \frac{CAPEX \times CRF + C_{\text{OM}}}{E_{\text{annual}}}$$

where the Capital Recovery Factor (CRF) accounts for discount rate $r$ and project lifetime $n$:

$$CRF = \frac{r(1+r)^n}{(1+r)^n - 1}$$

Capital costs are estimated through component-level scaling relationships [@Adiputra2020; @SotoCalvo2025; @Thirugnana2021]:

$$CAPEX = \sum_j C_{j,ref} \left(\frac{S_j}{S_{j,ref}}\right)^{\alpha_j}$$

with scaling exponents $\alpha_j < 1$ reflecting economies of scale.

## Uncertainty Quantification

Monte Carlo analysis with Latin Hypercube Sampling (LHS) propagates parametric uncertainties [@Shields2016]:

$$LCOE_k = f(\mathbf{X}_k), \quad \mathbf{X}_k \sim LHS(\mathbf{X}), \quad k = 1, \ldots, N$$

Sobol first-order sensitivity indices decompose output variance [@Azzini2021; @Zhou2025]:

$$S_i = \frac{Var_{X_i}[E_{\mathbf{X}_{\sim i}}(Y | X_i)]}{Var(Y)}$$

Parameters with total-order index $S_{T_i} > 0.05$ are deemed influential and warrant focused research or risk mitigation.

# Figures

The framework includes visualization tools for analyzing thermodynamic performance and economic metrics across configurations and spatial domains.

![General process flowchart of the OTEX computational framework for OTEC plant design, simulation, and technoeconomic assessment.\label{fig:flowchart}](figs/Figure_1.jpg)

![Kernel density estimate (KDE) distributions of (a) net power output and (b) levelized cost of electricity (LCOE) for nominal 100 MW gross power OTEC plants across the evaluated domain in Cuban waters. Solid curves correspond to offshore configurations, while dashed curves represent onshore installations.\label{fig:kde}](figs/Figure_2.png)

![Median net power output and LCOE across Cuban waters for nominal 100 MW gross OTEC plants (low-cost scenario), grouped by thermodynamic cycle (a--b), working fluid (c--d), and Pareto frontier (e) identifying optimal configurations.\label{fig:pareto}](figs/Figure_3.png)

![Spatial distribution of median net power output (MW) and levelized cost of electricity (LCOE, ¢/kWh) across Cuban waters for nominal 100 MW gross OTEC plants under the low-cost scenario, comparing (a--c) Uehara cycle with ammonia-water mixture offshore and (b--d) closed Rankine cycle with isobutane offshore.\label{fig:spatial}](figs/Figure_4.png)

# AI usage disclosure

No generative AI tools were used in the development of this software, the writing
of this manuscript, or the preparation of supporting materials. Claude Code was employed to write the documentation with human supervision.

# Acknowledgements

We acknowledge the use of data from the Copernicus Marine Environment Monitoring Service (CMEMS). This work supports the advancement of sustainable marine energy planning.

# References
