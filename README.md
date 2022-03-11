# ROdemos

This repository contains a number of practical and tractable examples of robust optimization (RO), applied to problems as diverse as experimental design and multicommodity network flows. They are written in [Julia](https://julialang.org/), using the [JuMP](https://github.com/jump-dev/JuMP.jl) modeling environment. The models accept a variety of optimizers compatible with JuMP, but [Gurobi](https://www.gurobi.com/) is the default, available with a free academic license. 

They were initially created by me (Berk) in Spring 2021 when I TAed the 15.094 Robust Optimization class at MIT, taught jointly by Prof. Dimitris Bertsimas and Prof. Dick den Hertog. Now, the models exist here for educational purposes, and to demonstrate the potential of RO in solving real world problems with uncertainty effectively. 

Please use the included Julia environment (in Project.toml) to be able to run the examples seamlessly, and feel free to post issues if any problems arise. 
