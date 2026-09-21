# Main independent synthetic oracle for Task7 analysis

This is an algebraic test oracle, not experimental/model evidence.

Use a condition mask with two selected scalar coordinates, z0=(0,0), v0=(sqrt(2),0), so mask RMS(v0)=1. Let F0(z)=b0+M0*z, F1(z)=b1+M1*z, with arbitrary fixed offsets b0,b1 and

M0 = [[2,1],[1,3]], M1 = [[1,-1],[2,1]].

For delta0=(c,0), actual delta1=(2c,c) and delta2=(c,5c).
RMS(delta0)=abs(c)/sqrt(2), RMS(delta1)=abs(c)*sqrt(5/2), RMS(delta2)=abs(c)*sqrt(13).
Thus A1=sqrt(5), A2=sqrt(26), A2/A1=sqrt(26/5), regardless of sign and amplitude.

The second-step derivative ray is w=sign(c)*(2,1)*sqrt(2/5); it is NOT the initial v0. A central quotient at beta .1, .2 or .4 is exactly M1*w under real arithmetic. The prescribed prediction q_.1*RMS(delta1)=delta2, hence Eprop=0. One-sided slopes=1, R2=1, adjacent secant cosine=1 and relative RMS change=0. Floating tests should allow only their explicitly stated arithmetic tolerance, never relabel synthetic arithmetic as model exactness.

A deliberate erroneous implementation that projects delta1 to the old v0 predicts the wrong second-step response; this oracle catches it. Offsets ensure raw-output division or mixing z1/z0 cannot accidentally pass. Use separate baselines for each stage and include a cross-stage baseline tamper case.

For denominator convention, q_next=1.2*q_current gives relative RMS change .2 with current denominator, not 1/6; maintain Task5 convention. Zero floor means NONZERO_ABOVE_EXACT_ZERO only for nonzero response, not infinity in JSON. Two baseline samples cannot establish a calibrated confidence interval.
