

function minimize_plmsym_mb(x, Zmb, Wmb, var::PlmVar;opt=Descent())

    grad=zero(x)

    pls=pl_grad!(grad, x, Zmb, Wmb, var)

    x .-= apply!(opt, x, x̄)

end

