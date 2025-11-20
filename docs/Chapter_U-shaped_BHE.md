# 2.3 U-Shaped Borehole Heat Exchangers and Heat Transfer Dynamics

2.3.1 Technical Configuration and U-Tube Variants

U-shaped borehole heat exchangers (BHE) represent the most common design for vertical ground heat exchange systems. These configurations typically extend between 100 and 300 meters in depth, making them practical for installations where surface space is limited. The basic design involves two vertical pipes connected at the bottom, forming a U shape. Fluid descends through one leg, travels through the connecting section at depth, and returns to the surface through the other leg [1].

Several factors control system performance. Ground temperature at different depths, thermal conductivity of surrounding rock, pipe material selection, and flow rate all influence heat transfer efficiency [1]. When fluid flow increases, the system extracts more thermal energy, but return temperature may decrease. This trade-off requires careful consideration in applications where consistent heat supply is critical. In boreholes with layered geology, thermal interaction between descending and ascending flows can reduce overall efficiency. Increasing the distance between pipe legs, known as shank spacing, helps minimize this thermal short-circuiting [2].

The single U-tube represents the simplest and most widely deployed BHE design. Two pipes, commonly 32 mm to 45 mm in diameter, are installed in a borehole typically ranging from 110 mm to 140 mm. This configuration offers straightforward installation and reliable long-term operation. The primary limitation lies in the fixed spacing between pipe legs, which can create thermal interference when both pipes are positioned close together within the borehole [2]. Field measurements from operational systems show that single U-tube BHEs achieve borehole thermal resistance values between 0.10 and 0.15 K/(W/m), depending on grout thermal conductivity and pipe positioning. Systems using thermally enhanced grout materials reach the lower end of this range, improving heat transfer between pipes and surrounding rock [1]. Installation depth typically ranges from 150 to 300 meters, with deeper installations providing access to higher ground temperatures but requiring proportionally higher drilling costs.

Double U-tube systems incorporate four vertical pipes within a single borehole, effectively doubling the heat transfer surface area compared to single U-tube designs. This configuration reduces borehole thermal resistance by approximately 30% to 40%, allowing higher heat extraction or injection rates per borehole [2]. The additional pipes also provide operational flexibility, as the system can run in series or parallel flow arrangements depending on thermal load requirements. The main challenge with double U-tube installations lies in pipe positioning within the borehole. When all four pipes cluster near the borehole center, thermal interference between legs increases, reducing the thermal resistance benefit. Proper spacer placement during installation ensures pipes remain distributed around the borehole perimeter, maximizing contact with the surrounding grout and rock [2]. This configuration suits applications requiring higher thermal capacity per borehole, though installation complexity and material costs increase proportionally.

2.3.2 Elliptical U-Tube Configurations and Performance Enhancement

Recent developments in BHE design focus on improving borehole cross-sectional utilization. The MuoviEllipse, an elliptical U-shaped BHE with internal turbulence ribs, addresses limitations inherent in circular pipe designs [3]. Standard circular pipes, even when sized appropriately for the borehole diameter, leave significant unused space. The elliptical cross-section increases the pipe-to-grout contact perimeter while maintaining structural integrity.

Computational studies demonstrate that elliptical BHEs reduce borehole thermal resistance by 17% to 18.5% compared to equivalent circular U-tube configurations [3, 4]. This improvement stems from two factors. First, the elliptical shape provides greater surface area contact with surrounding grout. Second, the geometry naturally positions the flow paths farther apart, reducing thermal short-circuiting between descending and ascending flows [4]. Field validation through thermal response testing in 140 mm diameter, 102 meter deep boreholes confirmed these simulation results, showing pressure drop reductions of approximately 32% alongside the thermal resistance improvements [4].

Analysis of shallow versus deep installations reveals that thermal resistance benefits are more pronounced in systems less than 150 meters deep. In these configurations, elliptical BHEs increased heat flux per unit length by up to 45% during initial operation periods [5]. A system-level evaluation showed that replacing circular U-tubes with elliptical designs reduced required borehole quantity by 6.7%, while cutting water pump energy consumption by 37% due to lower pressure drops [5].

2.3.3 Design Considerations and Performance Validation

Selecting appropriate BHE configuration depends on site-specific requirements and constraints. Single U-tube systems offer the most economical solution for installations where moderate thermal loads allow adequate spacing between boreholes. Double U-tube configurations suit higher-density applications or retrofit scenarios where drilling additional boreholes is impractical [2]. Elliptical designs provide performance improvements that can justify higher material costs in space-constrained installations or when minimizing borehole field size is critical [5].

Installation quality significantly affects performance regardless of configuration choice. Proper grouting ensures consistent thermal contact between pipes and surrounding rock. Air voids or poor grout placement create thermal barriers that negate benefits from advanced pipe geometries. Pipe material selection also matters. High-density polyethylene (HDPE) remains the standard choice, offering adequate thermal conductivity, resistance to ground chemistry, and durability over multi-decade operational lifetimes [1].

Flow rate optimization presents another key design parameter. Higher flow rates increase heat transfer coefficients within pipes but can reduce the temperature differential between supply and return, affecting heat pump efficiency. Lower flow rates allow greater temperature change but may induce laminar flow conditions that reduce heat transfer. Internal turbulence features, such as ribs in elliptical designs, help maintain turbulent flow at lower Reynolds numbers, improving heat transfer while minimizing pumping energy [3].

Operational data from multiple installations across different geological settings provide validation for design calculations. Systems in crystalline bedrock with thermal conductivity around 3.5 W/(m·K) demonstrate higher specific extraction rates compared to sedimentary formations with conductivity below 2.5 W/(m·K) [1]. Long-term monitoring also reveals that proper design prevents excessive ground temperature depletion. Systems designed with adequate borehole spacing and appropriate sizing maintain stable performance over 20-year operational periods [2]. Ongoing research explores additional optimization pathways. Variable-diameter boreholes, where the upper section uses larger diameter drilling to increase heat transfer in the most thermally active zone, show promise in simulation studies. Hybrid systems combining different U-tube configurations at various depths may offer performance improvements in stratified geological settings [5]. As installation techniques improve and materials advance, U-shaped BHE systems will likely see wider adoption in applications requiring reliable, efficient ground-coupled heat exchange.

These systems remain well-suited for buildings requiring consistent thermal energy supply throughout the year. Their self-contained operation requires minimal maintenance, and because they use closed-loop circulation, they avoid complications associated with groundwater extraction or reinjection [1]. As energy infrastructure transitions toward sustainable solutions, U-shaped BHE configurations provide proven technology for space heating, cooling, and thermal storage applications.

---

References

Recent work on borehole heat exchanger design shows progress in configuration optimization and operational forecasting. Elliptical U-tube geometries reduce borehole thermal resistance by 17% to 18.5% compared to circular designs through improved cross-sectional utilization and reduced thermal short-circuiting [2, 3]. Double U-tube configurations achieve 30% to 40% lower thermal resistance than single U-tubes by doubling heat transfer surface area within the same borehole [2]. These performance improvements translate to system-level benefits. Field validation confirms that elliptical designs reduce required borehole quantity by 6.7% while cutting pumping energy by 37% through lower pressure drops [5].

Machine learning approaches for temperature forecasting in geothermal systems have advanced substantially. Hybrid architectures combining convolutional and recurrent layers capture both spatial patterns and temporal dependencies in sensor data [1]. These models achieve prediction accuracy suitable for operational control when trained on sufficient field data [1]. However, model performance remains site-specific and requires validation across different geological settings and operational conditions. Input feature selection significantly affects forecast quality, as excessive variables can introduce noise rather than improve accuracy [1].

Field monitoring across operational borehole fields provides essential data for validating design calculations and refining performance models. Long-term observations reveal that proper spacing and sizing maintain stable thermal performance over multi-decade periods [2]. Systems in crystalline bedrock with higher thermal conductivity extract more energy per unit length than those in sedimentary formations [1]. Thermal response testing remains critical for determining ground conductivity and borehole resistance parameters needed for accurate system design [1].

Despite these advances, several gaps motivated the present study. Comparative performance data for different U-tube configurations under identical geological and operational conditions is limited. Most installations use single U-tube designs, leaving double U-tube and elliptical configurations less validated in operational settings [2, 5]. Short-term forecasting methods for outlet temperature have focused on generic architectures rather than configurations optimized for BHE system characteristics. The interaction between BHE geometry, flow conditions, and prediction accuracy across different configurations has not been systematically examined. Operational datasets large enough to train and validate deep learning models while capturing seasonal variation remain scarce [1].

This study addresses these gaps through analysis of an operational 128-well BHE field at 300 meter depth. Three U-tube configurations operate under comparable conditions. A CNN-LSTM architecture trained on 190,732 records provides 21-day outlet temperature forecasts. Results quantify configuration-specific thermal performance, validate machine learning forecast accuracy across different geometries, and demonstrate practical application of hybrid models for operational planning. The work extends current understanding by comparing circular single U-tube, circular double U-tube, and elliptical designs using consistent methodology and shared geological context.

```bibtex
@phdthesis{fadnes2025smart,
  author = {Fadnes, Fredrik S.},
  title = {Smart technology in brine-to-water heat pump systems: possibilities and practical challenges},
  school = {University of Stavanger, Faculty of Science and Technology, Department of Energy and Petroleum Engineering},
  year = {2025},
  address = {Stavanger},
  number = {869},
  type = {PhD thesis UiS}
}

@article{jahanbin2020thermal,
  author = {Jahanbin, A.},
  title = {Thermal performance of the vertical ground heat exchanger with a novel elliptical single U-tube},
  journal = {Geothermics},
  volume = {86},
  year = {2020}
}

@article{serageldin2018earth,
  author = {Serageldin, A. A. and Abdelrahman, A. K. and Ookawara, S.},
  title = {Earth-air heat exchanger thermal performance in Egyptian conditions: Experimental results, mathematical model, and Computational Fluid Dynamics simulation},
  journal = {Energy Conversion and Management},
  volume = {177},
  pages = {382--403},
  year = {2018}
}

@article{serageldin2021effectiveness,
  author = {Serageldin, A. A. and Radwan, A. and Sakata, K. and Katsura, T. and Nagano, K.},
  title = {The effectiveness of air circulation in improving the thermal performance of horizontal ground heat exchangers},
  journal = {Geothermics},
  volume = {96},
  year = {2021}
}

@article{alkdobi2023thermal,
  author = {Al-Kdobi, A. T. and Serageldin, A. A. and Sakata, K. and Katsura, T. and Nagano, K.},
  title = {Thermal performance enhancement of the elliptic and oval double U-tube ground heat exchangers under different configurations and operation conditions},
  journal = {Renewable Energy},
  volume = {213},
  pages = {471--490},
  year = {2023}
}
```
