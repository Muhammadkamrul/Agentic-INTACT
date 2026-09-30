metric,unit,availability,source,aggregation,formula,definition
"x_m,y_m",m,modelled,MobilityModel.groups[tid].xy,instantaneous,state variable integrated from velocity each slot,"UE position in the plane, origin at the serving gNB"
dist_m,m,derived,MobilityModel,instantaneous,sqrt(x^2 + y^2),2-D distance from the UE to the serving gNB
speed_mps,m/s,modelled,MobilityModel,instantaneous,|v|,UE scalar speed
los,flag,modelled,ChannelModel,instantaneous,"Bernoulli(P_LOS(d2D)) drawn at spawn, TR 38.901 UMa",1 if the UE has a line-of-sight link
pathloss_db,dB,modelled,ChannelModel.path_loss_db,instantaneous,TR 38.901 UMa LOS: 28+22log10(d3D)+20log10(fc); NLOS: 13.54+39.08log10(d3D)+20log10(fc)-0.6(hUT-1.5),large-scale propagation loss
shadow_db,dB,modelled,ChannelModel._update_shadow,instantaneous,"AR(1) with rho=exp(-dx/37 m), sigma 4 dB LOS / 6 dB NLOS",spatially correlated log-normal shadowing
fading_db,dB,modelled,ChannelModel._update_fading,instantaneous,"Rician (K=9 dB) if LOS else Rayleigh, AR(1) with rho~J0(2 pi fd Ts), fd = v fc / c",small-scale fading power
antenna_gain_db,dBi,modelled,ChannelModel.antenna_gain_db,instantaneous,"Gmax - min(12((theta-tilt)/HPBW)^2, SLA_V)","3D antenna pattern, vertical cut, with downtilt"
interf_dbm,dBm,modelled,ChannelModel.evaluate,instantaneous,sum over neighbour sites of load * Prx_per_PRB,inter-cell interference power per PRB
noise_dbm,dBm,modelled,ChannelModel,constant,-174 + 10log10(W_prb) + NF,thermal noise power in one PRB
snr_db,dB,derived,ChannelModel,instantaneous,10log10(Prx / N),signal to noise ratio
sinr_db,dB,derived,ChannelModel,instantaneous,10log10(Prx / (I + N)) with the subband-quality factor,signal to interference plus noise ratio
cqi,index 0-15,derived,channel.cqi_from_sinr_db,instantaneous,threshold table on wideband SINR (38.214 Tab 5.2.2.1-2),channel quality indicator reported by the UE
mcs,index 0-28,derived,channel.mcs_from_cqi,instantaneous,"min(2*CQI-2, mcs_<tenant> knob)",modulation and coding scheme after link adaptation
se_mcs,bit/s/Hz,derived,channel.se_from_mcs,instantaneous,Qm * R from 38.214 Table 5.1.3.1-1,ACHIEVABLE PHY spectral efficiency at the chosen MCS
se_shannon,bit/s/Hz,derived,ChannelModel,instantaneous,log2(1 + SINR),"THEORETICAL channel capacity; an upper bound, never achieved, reported separately on purpose"
bler,fraction,modelled,channel.bler_from_sinr,instantaneous,1/(1+exp((SINR - SINR_req(MCS))/1.1)),first-transmission block error rate
resid_bler,fraction,derived,ChannelModel,instantaneous,bler^(max_harq+1),residual error after HARQ retransmissions
harq_tx,count,derived,ChannelModel,instantaneous,(1-bler^(N+1))/(1-bler),expected transmissions per delivered block
offered_mbps,Mb/s,modelled,TrafficModel.arrivals_bits,mean over slots,arriving bits / slot duration,EXOGENOUS offered traffic; never depends on control
prb_alloc,PRB,modelled,RealisticRAN.step,mean over slots,"min(PF share * min(quota, cap), backlog need) * cell budget scale * (1 - reconfiguration loss)",PRBs actually allocated to the slice
delivered_mbps,Mb/s,modelled,RealisticRAN.step,mean over slots,"min(prb * W * se_mcs * (1-resid_bler)/harq_tx, backlog) / slot",DELIVERED application throughput (goodput)
phy_rate_mbps,Mb/s,derived,RealisticRAN.step,mean over slots,prb * W * se_mcs / slot,achievable PHY rate before HARQ and backlog limits
delay_ms,ms,derived,RealisticRAN.step,mean over slots,base_delay + 1000 * queue_bits / serve_rate (Little),mean packet delay
jitter_ms,ms,derived,RealisticRAN.step,std over 64-slot window,std(delay_ms over the rolling window),delay variation
delivery_pct,%,derived,RealisticRAN.step,mean over slots,100 * (1 - dropped_bits / offered_bits),fraction of offered traffic not dropped at the buffer
prb_util_pct,%,derived,RealisticRAN._assemble_kpm,mean over slots,"100 * (prb_used + prb_retx) / n_prb, capped at 100",fraction of the cell's PRBs actually used
offered_load_pct,%,derived,RealisticRAN._assemble_kpm,mean over slots,100 * PRB demand (incl. retx) / n_prb; MAY EXCEED 100,"PRB demand; congestion indicator, not a utilisation"
jain_throughput,index,derived,RealisticRAN._assemble_kpm,per epoch,(sum x)^2 / (n sum x^2),Jain fairness of per-UE delivered throughput
regret_throughput/delay/bler,fraction,derived,RealisticRAN._assemble_kpm,per epoch,"max((demand-achieved)/demand,0) and the delay/BLER duals",xSlice-style per-slice QoS regret components
margin g_i,dimensionless,derived,arbiter/margins.py,per epoch,"d_i (KPI - target) / |target|, clipped",normalised intent margin; the arbiter's only currency
reconfig_prb,PRB,modelled,RealisticRAN.step,sum over slots,allocated PRBs * reconfig_loss_frac during the transient,capacity lost to control-plane reconfiguration
handover_count,count,not_modelled,-,-,-,"single-cell association is assumed; UEs never change serving cell, so a handover counter would be identically zero and is therefore not reported"
uplink_metrics,-,not_modelled,-,-,-,only the downlink is scheduled; every uplink column would be a fabrication
per_subband_cqi,-,not_modelled,-,-,-,the channel is wideband with a scalar subband-quality factor; per-subband CQI is not resolved
