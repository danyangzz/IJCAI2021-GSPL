function L0 = Lss(A)
    S0 = A;
    S10 = (S0+S0')/2;
    D10 = diag(sum(S10));
    L0 = D10 - S10;
end