export toy_problem!

# Right-hand-side of the linear problem dx/dt = Aop x
function toy_problem!(du,u,p,t)
    Aop = p
    mul!(du, Aop, u)
end