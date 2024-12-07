def solveSpike(spike: HaloFeedback.DistributionFunction, e0: float, a0: float, t0: float = 0, N = 100, dfmaxAllowed = 1e-2, verboseStep = 100, hasDF = True, hasAcc = True, hasDFF = True, hasAccF = True) -> tuple:
    eccentricities = [e0]; smaxes = [a0]; t = [t0]; dt = 0
    m1 = spike.m1; m2 = spike.m2
    r_isco = getRisco(m1)
    
    dEdt_DF, dLdt_DF = averageDFLossRates(spike, a0, e0) if hasDF else (0, 0)
    dEdt_Acc, dLdt_Acc, dm2dt = averageAccLossRates(spike, a0, e0) if hasAcc else (0, 0, 0)
    
    dEdts = [dEdt_DF]; dLdts = [dLdt_DF]
    dEdtsAcc = [dEdt_Acc]; dLdtsAcc = [dLdt_Acc]
    ms = [spike.m2]; dm2dts = [dm2dt]
    
    dEdt = -dEdt_GW(m1, m2, a0, e0) -dEdt_DF -dEdt_Acc; dLdt = -dLdt_GW(m1, m2, a0, e0) -dLdt_DF -dLdt_Acc
    dadt, dedt = getOrbitUpdate(dEdt, dLdt, a0, e0, m1, m2, dm2dt)
    
    t_feps = []
    printR = np.logspace(np.log10(r_isco), np.log10(a0), 200)
    printR = np.flip(printR)
    printI = 0
    
    print("Simulating orbit evolution in a dynamic halo")
    for i in tqdm(range(N)):
        e = eccentricities[-1]; a = smaxes[-1] # _, [pc]
        dEdt_DF = dEdts[-1]; dLdt_DF = dLdts[-1]
        dEdt_Acc = dEdtsAcc[-1]; dLdt_Acc = dLdtsAcc[-1]
        dm2dt = dm2dts[-1] # [Mo/pc]
        
        if (verboseStep == -1 and printR[printI] >= a) or (verboseStep > 1 and i % verboseStep == 0):
            printI += 1
            t_feps.append(np.append([t[-1], e, a, spike.m2], spike.f_eps.copy()))
            print(f"SMA: {a/r_isco:.5f}, Ecc: {e} |\t SR: {a *(1 -e**2)/r_isco:.5f} ISCOs, m2: {spike.m2} Mo, Timestep: {dt/getPeriodFromDistance(a, m1 +m2) :.2f} Orbits") # Progress printing.
        
        # =========== Update motion ============
        dEdt = -dEdt_GW(m1, m2, a, e) *0 -dEdt_DF -dEdt_Acc; dLdt = -dLdt_GW(m1, m2, a, e) *0 -dLdt_DF -dLdt_Acc
        dadt, dedt = getOrbitUpdate(dEdt, dLdt, a, e, m1, m2, dm2dt)
        # ========= Choose a time step =========
        
        dt_a = 1e-3 *a /np.abs(dadt)
        dt = min(1e-3 *e /np.abs(dedt), dt_a) if e > 0 else dt_a # [s]
        # dt *= 1e-3 if i < 2 else 1
        
        # ===== Update Spike Distribution ======
        T = getPeriodFromDistance(a, m1 +m2)
        theta = np.linspace(0, np.pi, getGridSizeForEccentricity(e))
        r = getSeparation(a, e, theta); u = getOrbitalVelocity(a, e, theta, m1 +m2) / 1000 # [km/s]
        
        # dfs = np.array([spike.df(_r, v_orb = _u, v_cut = _u) for _r, _u  in zip(r, u)]).T
        # dfdt = simpson(dfs, theta/np.pi) / T if e > 0 else dfs.T[0] / T
        # dfs = -np.array([df_acc(spike, _r, _u *1000) for _r, _u  in zip(r, u)]).T
        # dfs = np.array([spike.df(_r, v_orb = _u, v_cut = _u)-df_acc(spike, _r, _u *1000) for _r, _u  in zip(r, u)]).T
        dfs = np.array([(-df_acc(spike, _r, _u *1000) if hasAccF else 0) +(spike.df(_r, v_orb = _u, v_cut = _u) if hasDFF else 0) for _r, _u  in zip(r, u)]).T
        dfdt = simpson(dfs, theta/np.pi) / T if e > 0 else dfs.T[0] / T
        # dfdt = 0
        
        # ==== Recalibrate step if maximum change is violated.
        dfmax = np.max( (np.abs(dfdt *dt)/spike.f_eps) )
        if dfmax > dfmaxAllowed:
            dt *= 0.95 *dfmaxAllowed /dfmax
        
        # ========== 2nd Rugne-Kutta term =======
        a += 2/3 *dadt *dt
        e += 2/3 *dedt *dt
        spike.f_eps += 2/3 *dfdt *dt
        # spike.f_eps = spike.f_eps.clip(1e-1)
        spike.m2 += 2/3 *dm2dt *dt
        
        m2 = spike.m2
        # =========== Update motion ============
        
        dEdt_DF, dLdt_DF = averageDFLossRates(spike, a, e) if hasDF else (0, 0)
        dEdt_Acc, dLdt_Acc, dm2dt2 = averageAccLossRates(spike, a, e) if hasAcc else (0, 0, 0)
        
        dEdt = -dEdt_GW(m1, m2, a, e) *0 -dEdt_DF -dEdt_Acc ; dLdt = -dLdt_GW(m1, m2, a, e) *0 -dLdt_DF -dLdt_Acc
        dadt2, dedt2 = getOrbitUpdate(dEdt, dLdt, a, e, m1, m2, dm2dt2)
        
        # ===== Update Spike Distribution ======
        T = getPeriodFromDistance(a, m1 +m2)
        r = getSeparation(a, e, theta); u = getOrbitalVelocity(a, e, theta, m1 +m2) / 1000 # [km/s]
        
        dfs = np.array([(-df_acc(spike, _r, _u *1000) if hasAccF else 0) +(spike.df(_r, v_orb = _u, v_cut = _u) if hasDFF else 0) for _r, _u  in zip(r, u)]).T
        dfdt2 = simpson(dfs, theta/np.pi) / T if e > 0 else dfs.T[0] / T
        # dfdt2 = 0
        
        # ============
        a += 1/12 * (9 *dadt2 - 5 *dadt) *dt
        e += 1/12 *(9 *dedt2 - 5 *dedt) *dt
        spike.f_eps += 1/12 *(9 *dfdt2 - 5 *dfdt) *dt
        # spike.f_eps = spike.f_eps.clip(1e-1)
        spike.m2 += 1/12 *(9 *dm2dt2 - 5 *dm2dt) *dt
        
        m2 = spike.m2
        # ========== Save the changes ==========
        t.append(t[-1] +dt); smaxes.append(a); eccentricities.append(e)
        
        dEdt_DF, dLdt_DF = averageDFLossRates(spike, a, e) if hasDF else (0, 0)
        dEdt_Acc, dLdt_Acc, dm2dt2 = averageAccLossRates(spike, a, e) if hasAcc else (0, 0, 0)
        
        dEdts.append(dEdt_DF); dLdts.append(dLdt_DF)
        dEdtsAcc.append(dEdt_Acc); dLdtsAcc.append(dLdt_Acc)
        ms.append(spike.m2); dm2dts.append(dm2dt2)
        
        if a <= r_isco: break # TODO: Check the correct condition for eccentric orbits.
    
    if a <= r_isco:
        # Force final point to be at R_ISCO.
        e = np.array(eccentricities)[:-1]; a = np.array(smaxes)[:-1]; t = np.array(t)[:-1]
        dEdts = np.array(dEdts)[:-1]; dLdts = np.array(dLdts)[:-1]
        dEdtsAcc = np.array(dEdtsAcc)[:-1]; dLdtsAcc = np.array(dLdtsAcc)[:-1]
        ms = np.array(ms)[:-1]; dm2dts = np.array(dm2dts)[:-1];
        
        t = np.append(t, interp1d(a, t, fill_value = "extrapolate", bounds_error = False)(getRisco(m1)))
        e = np.append(e, interp1d(a, e, fill_value = "extrapolate", bounds_error = False)(getRisco(m1)))
        dEdts = np.append(dEdts, interp1d(a, dEdts, fill_value = "extrapolate", bounds_error = False)(getRisco(m1)))
        dLdts = np.append(dLdts, interp1d(a, dLdts, fill_value = "extrapolate", bounds_error = False)(getRisco(m1)))
        ms = np.append(ms, interp1d(a, ms, fill_value = "extrapolate", bounds_error = False)(getRisco(m1)))
        dm2dts = np.append(dm2dts, interp1d(a, dm2dts, fill_value = "extrapolate", bounds_error = False)(getRisco(m1)))
        dEdtsAcc = np.append(dEdtsAcc, interp1d(a, dEdtsAcc, fill_value = "extrapolate", bounds_error = False)(getRisco(m1)))
        dLdtsAcc = np.append(dLdtsAcc, interp1d(a, dLdtsAcc, fill_value = "extrapolate", bounds_error = False)(getRisco(m1)))

        a = np.append(a, getRisco(m1))

        t_feps.append(np.append([t[-1], e[-1], a[-1], spike.m2], spike.f_eps.copy()))
    else:
        e = np.array(eccentricities); a = np.array(smaxes); t = np.array(t)
        dEdts = np.array(dEdts); dLdts = np.array(dLdts)
        dEdtsAcc = np.array(dEdtsAcc); dLdtsAcc = np.array(dLdtsAcc)
        dm2dts = np.array(dm2dts)
        ms = np.array(ms)
        
        # Add the final spike DF to the data
        t_feps.append(np.append([t[-1], e[-1], a[-1], spike.m2], spike.f_eps.copy()))
    
    # print(t.shape, a.shape, dEdtsAcc.shape, dm2dts.shape)
    data = np.vstack([t, e, a, dEdts, dLdts, ms, dm2dts, dEdtsAcc, dLdtsAcc])
    t_feps = np.vstack(t_feps) # t = t_feps[:, 0]; feps = t_feps[:, 1:]
    
    return data, t_feps

def continueSimulation(spike, data, tfeps, N, dfmaxAllowed, verboseStep, hasDF = True, hasAcc = True, hasDFF = True, hasAccF = True):
    # Fetch the last values from the previous simulation and use as initial here.
    t0, e0, a0, *_ = data[:, -1]
    spike.f_eps = tfeps[-1, 4:].astype(float)
    spike.m2 = tfeps[-1, 3].astype(float)
    data_, tfeps_ = solveSpike(spike, e0 = e0, a0 = a0, t0 = t0, N = N, dfmaxAllowed = dfmaxAllowed, verboseStep = verboseStep, hasDF = hasDF, hasAcc = hasAcc, hasDFF = hasDFF, hasAccF = hasAccF)

    # Combine the previous and new data.
    data = np.vstack([data.T, data_.T[1:]]).T
    tfeps = np.vstack([tfeps, tfeps_[1:]])
    
    return data, tfeps