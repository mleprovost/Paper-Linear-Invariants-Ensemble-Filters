export spectral_linear_advection!

# Right-hand-side of the linear advection problem formulated in the spectral domain
function spectral_linear_advection!(dû, û, p, t)
    c∂x= p["c∂x"]
    dû .= -c∂x .* û
end