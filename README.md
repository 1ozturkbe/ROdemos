# ROdemos

This repository contains a number of practical and tractable examples of robust optimization (RO), applied to problems as diverse as experimental design and multicommodity network flows. They are written in [Julia](https://julialang.org/), using the [JuMP](https://github.com/jump-dev/JuMP.jl) modeling environment. The models accept a variety of optimizers compatible with JuMP, but [Gurobi](https://www.gurobi.com/) is the default, available with a free academic license. 

The ```demos``` are a work in progress, and are inspired by the ```homework``` and ```recitation``` components I (Berk) wrote in Spring 2021 when I TAed the 15.094 Robust Optimization class at MIT, taught jointly by Prof. Dimitris Bertsimas and Prof. Dick den Hertog. Now, the models exist here for educational purposes, and to demonstrate the potential of RO in solving real world problems with uncertainty effectively.

Please use the included Julia environments (```Project.tomls```) in each folder to be able to run the examples seamlessly, and feel free to change the default solver to a MILP solver of your choice. Note that the demos/homeworks/recitations use Julia 1.6.5+ (the long-term-support release forward). Feel free to post issues if any problems arise. 
