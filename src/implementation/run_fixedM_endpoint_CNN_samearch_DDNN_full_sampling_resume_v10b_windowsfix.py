# -*- coding: utf-8 -*-
"""
run_fixedM_endpoint_CNN_samearch_DDNN_full_sampling_resume_v10.py

One-file unified workflow for fixed-M endpoint baselines.

CNN
---
Forward direct network:
    h(0,t) -> h(4L_D,t)
Inverse direct network:
    h(4L_D,t) -> h(0,t)
The forward and inverse CNNs are trained independently.

Same-architecture DDNN
----------------------
Forward network exactly reuses the formal Fourier-PINN ConditionalPINN
architecture and encoding:
    (z,t,A1,...,AM) -> (u,v)
It is trained using supervised endpoint labels at z=0 and z=4L_D only,
without NLSE/PDE residuals and without intermediate-plane labels.

DDNN inverse comparison does not train an unrelated inverse network. It freezes
the trained forward DDNN and optimizes the amplitude vector from the terminal
field, matching the inverse principle used by the PINN. This is necessary for a
controlled physics-vs-data comparison.

DDNN training randomly resamples endpoint time coordinates every epoch, while
DDNN validation uses the complete stored endpoint time grid by default. This
avoids duplicating time coordinates (for example, requesting 7500 points from a
1365-point grid) and keeps model selection based on complete endpoint waveforms.
All validation and unseen evaluation paths are batched. Python 3.8 compatible.
"""
from __future__ import annotations

import argparse
import copy
import math
import re
import base64
import gc
import json
import sys
import types
import zlib
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

SCRIPT_VERSION = "ENDPOINT_CNN_SAMEARCH_DDNN_FULL_SAMPLING_RESUME_V10_20260704"
_CNN_SOURCE_B85 = 'c-rlKYjY&Wap3p)741=M#0<A*7Qikk-k8HUaDgTF3}82aUFsY#uA?#209rfKJ?fs>UC8EzWLpwxks|H0WQRqaC7)#LB$4v@MM@N<AL9j<{N%rInOTpnSNHT_0qS8fY%Qj{Dl02HD_@nB^(n9ORL2{n!#FwYd9&$3=UIHxXf~UTY?cHE@xy4i8AQo&lE%q2xW2U&+(>6x9A#^3@b`W^jI(GkjngC?1<@oO90m8Ub|=S;#>qE-dh+hyc$?niAO869AOGa^pa1UU!@oOu^YxRr{@epF-W87n>L8^5_}Sky8mI64A9#y@dhp*-YssIsz0ULS*EMe=Sc6}U#>ZdyverhtVeOWv&FQz_Kl#O<JpSq5pML9a8y)^%W4Y@+djD$@t=l&NiANv);_+X8|K#1*Prmg1$KQPa<gM2q|IMGCy!THZ{lgcpoWA`7_~+y&Upo2e-#`A~Pfy-@Rg4P%4Z~@eN7E~L6eYnR9q-3Uh*ZeCgZ#ePXqLE}t#rN9AHVtN{U5LqKmO7ik3W3(<jo(PzWLUppMUew`(J=z8TH`R$&Y_>`t^T2edApPi;sT(zfQjNJ!o?B-8UG4z02O|*S>f9m;a_dY+S{uzriN^=>2y-{;Pj^^s{fD{>3||um0fUZ~fh)58r<D{!dT8{gub>{QTd)@W+j7N^5UxV`u%!?X`99^xdzXy!Y<Ef8mRdU;Dq0-}>_D8-IK9-TwzaPyXz0AAjvDC-1&``o+IsBtL!i8;rVxqiArC7ze>V-KFpT_?!O%7(V*odqDNmum8*GPv25e1w`1Su*Wl9jYs%&PhS7h$y+~EsL=78p#I}$-z-`%I-ma4zroy3zV#MN)|-0pt;GP|<JW$2^7Aj^sGp{<^7tR#IQ<52?bo3Z<I9u3c%3od$*W&JdG&`Mz5CAN5B}uwSAX>QhhKgC;h#VL;BD;O`1pgr{^;HBe)N?eoc!7MPu}|;(Dw9SzbKiIv*O2p{cor5{;RhPlQ@0-HILSf#{izb_JxnX@ERN1+w8CW&u0-$V9$^L<ogVd)9-)h<bQt^XUD;;0GN+{{HKq8_K#=id=)#-9hpts>U^!%$ZBVz(MP{{>!a_y4vcvEjdxDo`WoYi)7Sn19>NL)zCZoecTWEN!$%){_vDR#K6&%U_^sh1zxPik@BH=2+uz48@;0oER~bk~>yO_1=Z}8z6~v9T1de$1EozCY`1ITFoxc4ehN|6u98I%$kO%oRi^4H3O%FcH>1#hZ{n2+$U;pCc58iq7i*Lc$ggje>r{DS>uv;Nqfa_3@y!TJ1Z@-3={@I(z?44(qJ=Xr?Z+=;D>~kPkpT9ClK)TLGqhOMylkkwFLU4cOiYZtoqv&BU4kvkVFiOMe)s-%^0O|VX4?g<w2PbcS<MEGv<Sluyih(_WDnL%4AN>4}pS{jjy7%e-La#~2o&4g%$6u$l)wq2;JxUYrYWG<WhXDlkM^Tp~f8!ua$6f&VO=npY1YSIzq}kL9lO&xo`PFEMXW8K-%<@S5M!ClSJRFF>KAWeB_<I~qkHp_um<-dg_>~{crtwJpn#SXZAp*D^jpBWQ#cc#YFQ&&6&|Bj9YI5v*Yw=*}dmC{M|9fk~^pfxGM4y{Q$soeN;KiL8P*u(#nT;pMUYL8yL_C@RUhn|^Z!(nlq}kv|`_WAj>Xw*Ky9cv`>qKmDgA+6v<&i)Z#p>-a3&*+d!FL^M${|1^Hm2Eej~GTk&%#)0<P!G5;r4-l8qvc+G?{wq^bfql7x0V$1VDhv_AEgfuV-1BwS=Vfn%C(oFES{o`>xk4Alx2>gUFj6#krRcvKY{a6Zjc<c{q+d==)i)7`!Y^r(N$>79Yk)rwGyLd6%-LcL^~iaG=W@JJ+{w-o6vOyuQ72^VXKv_nPZlYqxLR+`7Z8kz04SZ?10#WRARWbM5B#`t>`|aCIYCzkTcaOTo)mgO#O~XO=#_w9;&BuHSj-*4hq0X*FRsP2Xz{hwyK^Av>~{2Wj>o%!b$?PVPs*;7vs4Q(pf({{wh0MoE;x{E_wJX&mL}K1O309e5y80uU*~7XD8UJ_xgzw$daLJb_EGOKspSwQINORWj*Tyk$&34wD&JfUXAd0dF&y4a07n2jTrN9)%!nTJ4_X8l{N>D+os;6&@Q=ER3SxemJ-XMrw`#VEqlFX_SrQ1jIl*fbV%Hn?=<<_CY)xjl=97w7wC-LZ#;BDk_GZggMASM%wTo3Wm}Bco4Ow(Zgxa13g02Wh1dyoXRMO@@X`LVW1VYmsFB3yZJQCrul<-dIW3=Bx$xm_6|K<mo-Q^$&g~rU0~n@<l~-q{qEYTH%nyW-}PoW3eM}d?{=GQ*)NMg+$S2=F#Kf3o48&}l-rybpg@NKiu)Ey?H&nhq6G@(U6`_ua(b_)Ld51>6oNT*>Jkj3Obb2=xhDN4xhBoF80JA1Pa|_2-@6}<W>L==%V2aG0#sI%OuOTI;J&ojkGzju;=?xwt_jk6^s`-bVM7vHl46AU5cEghqM?0QSVMr-UjaBtj><F42XWjdmTLR5qd75OiaVSg>VlLFMSu3B0^21cfK3S|sFAVxyUp%r!Pss!FCC%~rKSw8wg>WzQ3qu)0TqLVlL@SYAWI+Q8ZGfbk2LA7PEV7dz`4kn)ZXBw##t#&k8)Tf)R+Bgu|+^nV1Z0qZTKh1<IhLl^WG903YQFPk;1H6%@7255<M8jNz`vPeUt8;JU+y=m3v13dK4A-)7Q=b0Z_$Vod0L&LGkIy`=n03_Us^zM#BUYUEUu7)-5E<?$TcOUUUpIP*|W?cbtH1mOTv`Lxb_&5iAS9pxto2q|b-{qwo+IIj}MVU`bf2xw#<0650hhv8R1Ro!r1PE^7Dt-jeyHQNgra^7$Y`r4Ju+$ksK`lQ8i?21SE(l=pX=@$g}DukC`U5d)<fV`rV>Dr?hl9Ab9}W=awQ=F^nEMcW{Lxp>zn#R&@bQZPZA4Yf<&8mGf()C2RcyO|DWqll;qIBJU$ah6c?*|;SEbiwY7MlH59<x`vk0q!azaYZUH8He{G_^R*<gD&|5>;^VQf;uxH?aufN$Py2knFfR*leEX=0((z=18EZY4VnX)$Gf2Q!3fT>5UlFrZ_X~@CG22dFHJ_rWZR{}kV6!JSjj<rILnwlOH-s-!P^lF7j@Cm;GBm<Vz{@x=e(=RXCOoMW&F5?nMz2kPQ6hS=2P#gb5syAq*LPzRBBWC_MBJ+b$~y|;ll#@a<NPiw+uf0uqEZGNz%6WlsW~m$lK0gKt4;@>N|ZzL5+^LB2@zs1hjTvG01=&!r5pVz;k9M$q&kTW&(E5Y$`^FW?kh;OU1fjatxOnY`h$tfdX*_#%@oY#hx;Aavea{)&-=(>8XH(;bFk@9mookjJPb2J;ax4IT7nA{LQHCUM#bHVMVK}rRXb{#`U8#1-=bE-40Ed<;UQ7De*n6VTq9k=2<k#<=PN<0O=U_NtRQEc5A7-1nWjNey;B=cbD1~%c#lRIvPz9#~aUJDeaROZ7uoU@}4jbrQygvVVKi&TlNPTf=3?pp$W|FnQIcs5vXDb1DE(p6>f4>PamWX5>+m8Ii#G_^F$0-NXt=YdBu|Of+Hn*-9b7z77{lJ?nTiAW{nF40~cNR41l9Cp{I)BE_+Mer_J6dTfid3wXgT|c^lqg?&>Z$koQP4FbT+;Fqd{QNW(E*-VYZqj=v|B9C2=7e%Y2lucu(gJXdfBe4R<km(_ka9UPGjF=qkY#nm$!VZZ=XQSLoBidY2U!BGlaD0u<b(LoG`BCssjqildKlVs};sdMqgH~T9Nb~*y%@6%$Su54VO+JEL+@%Rn~HAHlW6E!DPB@A$o4kN%d<9OTv-50{b&JIDl=NNthwJCZc(kLG~G#;6+=?q^_ForTli&@a5Y9&hitdCHr6@6`LTE($M8Os+`Xj>~swr;2lhHMB-gubDTRDlc;))>K8Mxg04uSs`}(A4GoOw?8IW__NU&&m1G?BD=6$J<ZS5sw}KuS)LxeKhgC=zf$P>(U4eN)Q8J5T)R%37#ZmIWO#tkyUtz7x-p2O+l)}19*B2k{4|Di27iVjWe^i!8oVrltn6*Lvo5p@8kSZg`I$(!A;|Q=>g-SJv;$1sj(E9&biJ{WdfZGN5M2mpBBf1IcLa53}73avSo#7fG>DU(SVVJc@!ifmwl7z*hpwSQbYtv3ZLvSLiMB!<$DH;9@7_MZ+EF{j{&$$E%z=YlDEz`NjsA;X5Y)p(^)o%&Of3z7=?K*2fPjA?Ev>il0&z_(<)5!IGN0*^m#zG;GbeU7epC_(V-2{7<kj$*}1VPSdFX@f`|(*0Eq&oWuSO7(da-~v$G@s*PcfYibg~M7CBBwpmZoYR=gjCz^4E@0B46SIKJNX7=swE%ZEXQ$Nnnh8^b!lxH5)tHCMrGjQ?Gs2%Lpr+i29}fSV4_9sIKpUkB*X!k`)s&{e$eVxXw0p(?uQc&Y$QJSumDEcZsS4>c?`?;hXUZ}boZGth3tqi8Fg-at>t;ffL8-C#U~o4kA;D0Iegj={z*DC7wk<zPJjfkFz|+4C;ti@n&MC08;qse$pmvh9$AILoJ9yYnTlrBTH%$e@3Jg_^{Fq8Bot40URt+IollD4d{xo1#;_csxTJbhnAK4G*J$pf>kPKDs^(3d#bNw)Z=I@0!*C#jAY3yS#@6dPRekJp@s-aPmu;&rV-XaDvCD<1r%bB8U5+`oU51SA5)@+Veffmp=ryy_92&Q<<1w{o!&@vXS(ow5A%3yN71jiH4-GUO+%#?L7>_{ha*frf;<ccf&VODUJD|X3Y;!vPP`}&2~X?!WyhGaPbx<^UZ`DWEPV22vm>LtpHw0G*h<G-+Gf^9F0L3_2h0V29DUXJ?MoZ+~`1SG9baP_Yxr{cQ;UZwR-8@rr`{?S{MY}XN3nXeYKDkqv|1pp{X-_FBK<mDIon<6gyEiJ(lW{MOEmV>Ty9MHmIztrM(JhxEL0KcC`u`TVps0uKo@{ye|ZPS%4?oKvGtMFr7M1>n1cJ9R^lx(~@sG-Pxyec{$F7yQ{(8v)84xWl01G7sy_k5jY(>CWl#!E;eqg(>_&z4_c}{qRXDZ$NyvaQzwepB6?tif%wSpwGcwPdb;qyw~*FZ7ezZXBbiR<Fh*HeQrv+~9hxOTJcQ+bY3L18@CwL$AlH=i?2vO}|I*N{R0Tr5icY(P&?2rE&I(pyQ&-uD%NKRps^CErL$?!6(Zs0A{x0F6!VZ|jHCI>c78uy>H#3JOprz7RR8%z~RHO@vlt>;)B7O2;vu~ur_!pf#w%(b4xV^f`{8O|u4nU}ab19-BGVtg$BrNfbv_}rnq&0c`=#m;z+u7K(9h6-|f`i;3<TqoDwz18v?+EPTm)s+oVQ{B#bU@U=#CwJ{(Y%EFi7+7_IUae{W9{{3!p>kqy@a<--Ki1EwDWP9yJZ?Z-tGC`(%xB^5gl~4ty;os-1D*Zf4fxQo(22P@+XeI;+2il>^K<jTZmg9rkm^^PorFGblD<+*{TgFw3e4vu638Zr=Idwl(8$AQm#f^fsC-|LL7rl)?GgEHeWEAPR_%eB#H%!l9V~)G8!iH25_fWi3N(JZN(d}yY<IMV+MS=ezxL|WiiNk?#snX9fHlLQekMQIP3}vTpX9<9BLp{O!b0svdyAO>8ujg#=U%2g%lCfaf`xwv#OYxP4QCMQu=Y8bYe~^`N_{rU3^kBV48<agLJ8B_IZ7xML7hKJ?VsiU3~_pQFqh-ZgMY4A0(;_V?M2Xpo*0Xi1T86i;@{GDz3X7<?(P9j;^Q4{TD{*;9jf9G3nI~$PSLeB!NY%Y<u8;_o57*nj0caDU8_Od-2#6g`G_>38^cGR}8Xnh`Cj`E>H<7BZ0!a0&0@yNr-_Aiz7+1@iI!hr2Ar)&L&&%TkHvl$+V3=10WIagVb6c+7UL{)vx<XLA-BHL~qGw+~@xq%8pO7ug%#nGq`a2D`%Mg%CGJ8B`b3&Lq;};gDAvf2f(;2)$~r3<f)oa%O0o^3btM_y2tw#>1X_bP*cP(tsC9LXf$gTY}XR)+6swXeOIi$E6%=RJ5-={xj@a_z5#h+EpSXCVUd3CL3lsH-I>lb?TEj<<<43G8vrb}S925(hdfUV=B65ArgVVU%geTiFfB`5^1!Osd~f*~9{_55xeUvI?L8Trs*4L57Io2qVweSkc*bL+6K#Wmy>v`@oh3_V#a<?iT2H)4oFcJQ8tkLo&l#SKGcXgkwX1=)6@_iq@ipEroylUesk5?e`vXG_q9MBCl8<DoJ!Agt3kJBrN@OCRjJw~QuUtxumGZB=Ttjs6XsTGIwI!G-X%tp%888*`1Vkk^k#SVUG=_6ifTq`lv?^A0G@=bHi-4M}Q@)5(;)%GNx!_%PpD`zpj)a3LJ=w#Hu;H|0r59w($yD6~HW<Z=0zo>SiWtru*@wJ+p$Mz94KrvL8M%fyR%ei=*$@wvMe>-dfXNe5ZZY{eNJq1AqHqrT&fy3cf2d~dB&|sdR@^fDzuH7uMT0b-+eX4!sR5^=O}E>1!2BPr-|H;5>p*t5)Uso0M&*&~F~Zvdcr2}jZ#QCI3)^zU7&QDqQ2RLr$=hw8WhL@-L5V_u8VyIj!r-uPB|Z{6lcDc?6?#F1!sfi!kXbZ%I^|)JMT26{WVUu&&E028{#fptu3)^2peSblQJAx-;xUhyvLQvkr!;(&Pe4=Yva=^h6Z(7zNRcRZEBbW2m6_H!4MDi^F1{B&jPw3-8$KI;s#f7ix265H20}PGyMOf{9^!cl-G$21Ns2~}k~Vw>^9GNrgj{`X(BHQDqTwAzz4_eXWpKbK;Ypzu-HgV3I;{N~`@y-HziK;FVF|t5X1zkACpYdH-5eUuY^W<SS2M2p=7SmvvREw_Z~>H(8ZKAhGeS?&@g<Prm!$$zMEo$Kmt3;H084yu#B#rS%93KIg183ezY8bB+kL@b@$Ybv0XBMa&jev=5=>-gxsbGfq9C+;$6vm@@_T!#s&Sw)#RSLcJXDS%Z=X#miKY+I?4CEtqG3Fk=Fl@bIEV*vL>X-?-4^&DiH6=Z!eh)KCR64yPwz!gQ<|B8gpOU5L1!^Hnu(ksZJSMx!l?lzMuD0@^GOCD439L!7f5n|jaY$J9$ksYA)bl>;&4xeXIla+PO~Bzv$o{HjbsMLIvA)dH<u9sO#5ungwZq5y}_rKSF9Ff3`z)!yGUd-04%ROYc|mPUoBpP7XTkE7{+5~Z(W>vlm+wj`MZ`HVMS28Y+s`we#zE`6|vQg14A5>QOFM%@Xd3Ef+>s5@5%dE>f6vHlE7jN`QVH$bwMoEnkDsLZR%P?wsn^^nNl^DMTar&TnGEJg9GzeW>Xv8*q3Ko2J>D6D*VvW8k@d5uq}?0OpP$RLI)wQkfxx!*l8U!T9psnFjv%=B)qicgvi%TIWp(m7Jz$T+f}E0YIT{LZn|%crkMqLU%T~->l=4l%HzF;=AH(Ht%<hplCL7fqynCCTW%XWaypfS0cg!mvEZEhBV^R{=h48BHhtZ|nYXplF8j7ltSdfe9W+;X1YKFh^HU93Xn^TXvgDY~n8x9#^^}gYy1Yqpvp|H(**LtmWR!$3z3+MOxDyLEo2$BLK`z(2o)eR+n$OAO8j;R{#Vh82ZX7-z=Q#{M2g$SXHZ$fd*4B_U6)%rw?Z+VxvfdWQty>Ot@%bnN&vV)LR={I#SH%HKOt#9v;9MW5I1$4apaITIASKQOIszfV?Un|0-$X+iCc5yIqv`bTEW6Y(7a*Dq-dAOg4%|k|VJFo>#|YIL=8jWU3vVDGJ6@nkeB#JC(WIOUoQRvUj~2xVp6}N!k^RBQpN=}u?+>Pi9pa#yVW?=x`dac4g<A5@Wx^XbCg0W)lE_P9!kb}y*;)dN1#vZyb||-a-7tM&^%ccO_h<2F7*GI~Q4W^iQ94x8lAIAz@EAd_h|*)-EiAV&2?vp0z(8mm-zW&f6Peghyburj(e94Y2UKH29B~tk`X1GbQr%ZP_-EFd0cd}ya{vR-?6Lm(CvGz9u-Q}1e^UW9nY84MaOPb+?c$W7O=ZKv>`=a^)0EpW&}X+em7<?Sk$_}tBX^}~Fin#=WDEn-k0`pn(ohp=(SK7nM)i6tH0M4e{jMO)UiqGlXsMEA&%I5Z%f<yYjHbpMWCwHJdoDT2R{+h?=iKAsGAc)AvRM^d$=a;Kdm;$NrzyNv{KS^tRZ9igFI6tkBjI-3TiQrx;dImgQ(3seH)N7lHJ#OtRW@}~b447zxcz2QQ?X*yP8HToQ?&=}KJIY}iLUl2HISOzqHFSi32W{)^Ui&-sv*phfQei%3O5>jVt!@W#bS*;r#dwxgw7Q1E{ybM7yhA)F6sv^W}|Y0vx6~&#tKG<;^WHZ3Ky8Ni`(;3a`Er~$;d6ca)(IG6H1)r<#E+2iWecEfZ{I<Sj(+R*>0y$;SP<>y3uhxAq{=6*=#SM!};9i=^M(GGIY+IbP%WFX_sjpb~dz3@jCQUyZQsO6Kx88%rEUzjI?cMW9ig#tCmvwU62S+>1m6Bc9k4ZyBe~fT_p@0{ls!@S4af4n=TdF6=K2eRWg<$8`M6kd}zA`1XBl=n7A-Da*2j~Ho=Or(NHN1UMeZhKjUK~&2u^-z>7U!P?=TSUAPaE9~L$9M=&y54~6^V3IMQuFxMM)`1nB-A0Fv@%M+Az<_n|TrV5UGKG7tPN94WnXn33)q^$zGE^TJCyU?zoAb<tys7#9Ns5!g}0I$Pm)28z=Y69W7PjM8jMj8VT3nd-n&SBXAe4}s<#O`1ejwb;YTjKLn2gxE7qW7a9|J*EOm6Lb@+bC2jvzZgVV&kY5&?#mf&?j-@uh7Sy!?H>cHq$CtrL<VqD*G@7ssG{lkR1(!_B?diC^f$yAEF#&*#~5!)ck<v#IM*$s&tyBQxI#oQngbZyy%*;{HWNMlam_7`7zc_U7Xm>jlWjikLlCG)oUU~&7Oi@L9(Vq13;n4@5D-JFUM!?-RQ9PuW(~cDX_=ICXdn>r7wBiB&tDUm!<KmD>S(dqw$)_TCw4ZdQOW8NQxi<heZh%r5cK|Gltw`8|uA+J#x(p7oTQHt@dDk6aA&}0p!b9czV~2cEEWhTUG=#f-k_cgQ&DEqX&rVD8pTUikzthTO6g5G!<`Dra(uWe-}q8k+U|JT^vE21muA1=z?B=$$+d!4RqKnw3tkMAv%%?8UMgjUE}<tiH|F5nX_3{qB2j5>D$}H_DBLd9b9*swiv&f$NNcO*MTf2TMk(-5?pWw)ddIM?qX@NL6|#AnQ*RW)}68p%-xor)l-Z6vY0I+f*muOVEuCZR+aWp`2g<yD@GXiJ5>a@Td1<Ywddv7br5S(1g!*jwe9rwd(kmnb3l^9rX72}3$6CN%dUfig}NcyQA@029|D(yN}SAE?Y6ViQ!1`=w({6~F88%AZq)<h_qY#wg|b+wv9|rir|lQ6Dw}0Do<`%Gi!Gr_fEgoQiKy1FsS|`PtUj-h&x!#f(#TXA7q77rw~-Pm>Xb6oz5H3!)J6N{()OCA<{Fi~sy8BUTQyplp~(}99g`JU@$E&lxLnqy2|%Bc+PTC?L5((A31|h%*rQo6V1&h?4_W4Ag=CSG?6~kKs|VmbhpUdL!?G9BWlY4ry0lcb;p|(pCF(u#Y+dH|xy+Oq?oTq$aURY2q`Jkeo`Bw;E8y#WXM=I)BV7Ge+z8J_b?!%649)OtXx5}_J!EU)$*P<&wMp0Wlw+(GPyZz#>{2p;Vj(`Es&SRSe;EvcG{bWZm?hZ3kws{U=OD9^zC4RrR@TvsC}lTIv3(pxpEL_n+k+#EsH8`HpbQ67h{sqc>MGsqP&}7q|FEiG?b`dwJg9E?QFve0@uQg+75sR)KC`5biGpt?_~6AsG(TQM5%4qn)JB&Rj{&-ro7Jt9**Gv&vG6S%tb8D+HKN!QRNTj5YxJ<@y4$MY_JP*3c*c7T;{)y&YU7oQ+BqhL4PvelVd&YSIgh@SVCa+qB2<Z4;f2zgYkpMf;sODAM95UIm6n6-1s?%Wt7RcnB4pBKLA0oQEu>U+E_4k4XvOF}(kI}RC|A+l`lh9wL%Xx!F#|^1-QT5(TzY37RF~qofC?dUgX5yM89>+u#WqSyUbCc@Uv>xS*#4_~&($xk$0%(v_ckr19EftsQ&C1pz3fSgd9#2q%&Y)mB)zcP1OqH615yK6K))N%$Z%jY#k-3z?}WwE=vRPZr`2{gXVVt#`A;jAqaY9lXO*8yri7CGInC(Gqqbhko?ZJm=OelUp$lZWO~PXwNCuwpN2?4qVvukWjM4!+;jczs+YL4E!`{bNg82iCV$is|I$e2^xD_R1Uxz57d?JOFG+^wT?wMGr(NaBaf%m_V#eRp^PP~|v&X^09o0=r~6u-s5F1c={Pr8EZ<=K;SI1a~=1~AQ=-EbfZ+X3s)g{%~iq<arKSpIa#?lTZI!SU+EsJP>XZf<g)w{l*L>MZQ8)@fI3rEB+XH(iR+T@{uw5xckh+>tA}4tyc>iVYVf2TSg_w9K;neg1DnVg5eK0!sg~OX9oh_4kX6G4`0I#cL?5bmWz4*_BqmBO>_xwlc$lci&1&Jhy`%<ND6aJdA1AsFjQ;SWPtSS|+{3$Jem2Z23BuPKm>Bt}EHdC?U{&b)15rj{C&=O7@=Xx-sP=%Im)RGCfea0!-Uu$LjcwE4j=^>J=Z3=G8ZAl@8>q_W@dyblpa!&a$}nsIO!}Vr%tAByDL#55X(P14Hsch+;8HM&}vu>b(z4!SF675HQ1=m-2v@o@cVwFGMU4(9^xO9D)Mw3O5{@1cn-28X}<L%N6%J$!qfXWfuj^E7GtLP54z|=p49O(+jnYl77=yx{0O%zkZNkDPho{&IO)qyRx{URnHlfh$)cAVG<ui`IOz0;h&HFx5<-calI9X)XX$k#uZIck1bKn+e%(0$wrlel@w-nZ=Thbi+8g%du+AGJF%I-2tx31Vob2}?3#{EE|i((0A;}twgP2m(}#{ja$-QCtQH<}u9mA9RPGfDfNW?cL5rI<+?}zOJ+E*tIDkuDXW9F<%eqI=#am;IUT4z$lKcEx<zRFW(67vtCoM9LhB2DK9LNleD9jk}OAY~WQB*RKSrSj_qySoEO{X)X%U(T!e2#SHnd*i#))Us<qFPxqQguqC<)vZD5^}1Rb{%M9<5oR@Fb|HRdJ*x1_#xg$KFHEMH>i=jmn<(Y!H%z5fgox!uO5e8WZ%wbws-hPU`Ndb?M+6r+*!@TsN+$+V)?<xfZP7aXGg6ipItA1)Vj8@27fm#9;c$PdMDPDbz5?D5+jMuAkwpl@8{zeP;QG;9PoBcZNx;dFoLS}P%X`JkxIk45OCGHp3B>h%j84RziwCC25$BohH+W<!bDZYgk`w?O35g7cDW*9Qg#iHQvmRfiqad5%q%bsw{o79*O(d>n|>8KdOn2@1|tmk)tyT64l79C+@<Gx{E~5SdhW#&k#=nkqE!ZD?vZ?QH~<G>gcE34XGBam4eg#l3rAkBotvtT+sE>w*)$$?v+*>G@OB#B(qSRjy;fdYyS2?&B|%D*M4^embHbm+N4n^xGl4uChE3l0Q(osu|Ko5rk#Rb5%B<vCuee^)!;pH;OA6U{oT60Wkm4NgT_0*URd&x6Td{yzF29Dy<J6cSl2thcEW9daMp3mr1mXxjnlZ$50*WjCypl0ZPr9qaaBN6@?Q(Uc98sovrBp)-g<%FtP-(?9PB4rH;j!9YeXhCQIV=f`(3*&ivOvD-Zo|KWXk+_UavKkHgtG!=IWb?GMZY<YRV8&0;*PX71I#wX%2X6Spx4Sgz=V!*b42l}vp_s#%UI0{9ixENz|#{4k(3IE#gt`c;)_{0+@W_Zk&38hA@cJhcrR;kDDGt~y%kxzG`(OpCB3S(yoCR`PT4X;X~}L~Ud3MP!k76zIIhao<syrm5ej-9@LQ>=9b2xL<;w$j6UU<orb~v|2<uRhhG?+kKUL!-VZz5bigE26^SUd>rT`XS*)s$vyH0nMiBc^rv#w}Kh-*;;Jv;%{6E!KG>LCQm{1t_+@E^RW`haC076<i=n^T&9knBo;d;Ht=apzuq82mx&IPYH?Di`~3bAe1Ld;=saekk`ki(gTGg(v_I3LuvZYu)8xm1GptyaGkS5H;F%=C$R4*iCGk%8{qd4lKKpwM7f67cq$9k=yg(8J@hbvI(x;w(4-Ti;|*@du6t{3In%o>2;1f)60rx|CF~K4Zy#kt7nA>@lb?#Af;w>9WtpY%qhhyx1}?~Z&NM+VKqNe>mDrKOC1W>gFPZu9~CF13#E^92203Xp#uwOd?BbJO)fAggqiLL4zV6}zPq%y+Z<(0y?nK~u0*9A2mH37m<aKfT1p9d*>V1$^bEq>a|o4Z5%fdeVrx^JOeiASeB~7973Apy?rF=IF?_<ZOS`pUoYCTXEN-d`qcRM;cG|4q)?4(ffjKiZ@>}x|fgI%w#{?u=rF{z)#d<}+p&`~KpQ4YFfT^gOwY828xYY9#oeDs94Soqn0}7Lctqa0~AIzHDg{_q8E(TTk<*wu{@Dn@YlGlO+yKMFDwYwnhT5U?2Z?zo+<`*y{MzBV1BcV#}aio>Kl_niFSm6MWsSr-fw5YBAjNLM8tHWn~+Vd#Ej6injkaO<);ou$>&T6~*fdfr9?iSLw?WoR`BwchA^X_02Pl)ZQoOjT2Ue#D;{1^ET52?Fq=^{)l(G=hal{X!T2~VR*8PZ~I6Hx3?YoT)|MXa*){E4D39H&r}xAH8CBc<0~u&(nct{mvn!zijMF%F<;30zXMK%-@yHWZB@(ISMjnvCfy*ftCe7J~JT_e`4#)H>v%V5uE-DLL+<kW)MK!c)r%4V2D6>~LewzE6>R061Yyo(t<r*&`G%0`m2xdDU07v(~rYcN%2nS5h0u_6*xL0U<O~3Yr!cDom^Fyr{PR`YZ^buL;z%Y=_);T_f1H4dD`7ykBCKGli)oG}#sA`p!i5WrvW6lOn!>X-9!-oRpdF)nzHw6Kwqjbe4Lqt-pZcQq;Pt;{AHJmCKS*e3DY6JFkPUqrFtkrEt}mbysb@)mG|JSGd@YxfUy2#&JmUyWkX!ljz|zcp_gziM)Pp26F?&FrH-T{Rnr{`K=R;5_?8zNlCTS)bveD8l$s-jT`#uL(I2zg+SVtNu+Sk{0=P8qPistJ2I?>_QkXYlg=Lo??9gWb@9LF+;N_c^Aot2kU-n0bsU|M=d73#3M2=6JClViAVmZB4cDWR$IcX4d(W(JJkcS?4ZmU=#BG8%<)VewB|^QZ<0k3AH+6~!?W!&Y-Q}i|lW}Z7ulQw^a(NKWStHk`OjAX@4C^MhjCeC?DY=>|AuzE&#metW^QOF)MhMM*k~~=2*Q`n5f}ypkfMtXzoUpVuPFQ7M*s-#zdpoD5LYY4=1)6hmDzy1CTa4!!ivc&p7~RLFD!YLktM08SDv^)Euy#eY=0XS$FeW6`rUX`|8=W&6g%b=P<k6tu4?DaJb1Ndy2Ao%1b$y+8cr$Ue=UvJ>iq*kffc~ZKHL_XQAI2T?NLXwNNPsKLJ_G!|V%jX190?+>uyaM|SvnJbNr}(8%Xxt{Qt*vCAF39ANn2X(KXE^$+F~Bf@}u&b8LX?s87j}K9#NcF9ue8b$xzj;&l}zqzIIJagKbWwH1uff_~3l32J@7>*ya~(%79@{x@(5Hk(6%g=)9jr;XS!quMGDbxRuI&d<h<Mo4+6!c0}8X!eF|2;n?V(oB6$!GLN7LuB%1l9)vj=IYWNWWZ@lHdEd@EqKSUP?pinvZ(wAo#g#V(TJ@xUYu~GQAQV2PIQq)A<S=^JH?lnF7AL$Cv^kp|be`?x@nKUdr={XjmV+b&KV3L3wqA-@k+%Op+b_x?VIZ1zM{Bu+R}MZn;5yD7o4W`#Xs8b?RKj@>lHKccNQ)5949)cFUd{E@?Tz0LcJAD|9bCV)_44}mi|cDmtrohvG#h$;-RA0l57uwry8cqIy}o+=CDF4y->iap^4WMC0`r__`g0BQc>xpRiOuDv%O{oqoDQ>@H73PRk=8qLBddKkQQBC0X!fvKh36dGFKQ(EX-7)zO4u^P*vR%fQ)=tk6J>ujQh!yW*r3P!5L&YcyJ^+C%XSKd4;4(rjwJ8q*amo_E%WLu_i+f{Cl7H?JD_hxeFdZ1`({oIp6ffut{mAJK2|B`jxU>=$GT;s_OG_zns$bf#w}I(N20%((WG;1b;=XVnm5yybO>cxO<I3@;)*3{wwF<Nt$h)Tx2Vgvr<e=FJC$UWoaq#^o@RP}X^J-}4}&KsW)a5VV!Hd)$3%~WPRXfOqRi@GM?$^yekAe~MSdbCh{zg)Bu%1Wv9HMlOBc(2$Z#>X!P^(d(1rs$0>>OLVD$2o0Mhlexr8Xsq+xdCB#ZJW17q?N%!*XbUHM+HNJ6C2Qy%7WCP%VY^-?6YV_=_Tq9hymNfHMcgawZ|vQFoibNDo+9DY(sh{Y6YX8v(9pD8PEmpboN?#57}o!qh(sdrcOBt6}z=wIqyJ-C>3yt4QN*-1`zapd8-4XV;@lNxrDot9UE%=sthsR0W+FAp>2NJ^3jHnQ_5DQ{mD%Txf0<PYWQMGCAznt?E#qzU?HJaf;xgDH6TBCPAf()N^OJ-j`aUAIYzNeU<&j|(p*=AqALsR9>I6r83UALV;@m2EPdRmpCf!?E2Ju^shRLnLUSn`uuD=WIw;ZOhKMH%sr#HuLagvIDz@QyN>@fxo1@zzfd^bgzX)^8($1{Xc1TpvkAdv>ZW&p?{kx_$!hrNZ+LM%g+~7Ex^UI2DRQYchF$`C(9t5ZyTRa9-&SM%qgjyhCcI)$t<i_+4_^oCR{X+@JZ7M`HopWKZCG#?qFHg;GEs7T3Ldl^l;GxLB%L6(}J3lQF>!>N!~}D42K1?9I7%M%5oga@*KnmuGr|Dm*!A_n41YOFUz3-H8%~w-n)@dTy|v{1lL}=vn4Hg=O(-9Ryw=8l4vT#pkpgub3^srcyuxG^U*fEGa=B4Q`si_oQ?L1&Gvcw;)>xgpYNR8^Y+HKqD@OPzRVNazRSz-8`TM3hOZ$Lx>%v1g-Vpv*`s2ZX#v#oT`H@lfOE@Xq5YZ#>ao-)_;LPqJ}S$5;5zISr97;1AEmWDUMby>*0g}Vz3V6_b&gx(u6<Q)M70>_xo)Ta`ZMyPTV)k{B$fLk?*06LX(O7Iqf|yoC&t`mm8TX-C8*pM`%T!EQlsj6S4X_v2nZ=YtKM~<7u>XbUcLW3m$<3K4dfNXrFRmYWp@j$eZI4zZL<_*O{|h87<hM;_2@NpuaTl5(ReaF4ygX6UV7H9hH?SLXV=l6por*2-Dio3DIgq-```E#3ud)qp2wm+H||ovy$D9g15_ti2Y%Sa8f&CRRKUWRVKB`~)cR#?Ao9TqQF8|fdRwa^&R|2EKYT@2{}#>}dx?(OZb@bNvI6hU8tcx=>CTGibIP2nuX**Lf%-7d561Y65*m;O{N-LxJ)t!t&lL!bVLrSQF&-H4R;1Q0BZ5(oj_}C)%|RNWC{|*mbhUUAs(}Xnv34b~_XJSa74~7Gral9btTen`BXK$`cS?9qXww=F(}Vs}bp`}ka>-n9tXXRgtk|AF?7$yBtko`jsI_~gylSSpsC@IOY8-LF!{6{K;}Me((XUDkX_q9hi?vz>jOFj@@~w^{t}UhXyf%yivQAmoc3BBkS4BAmHm~}sjW(zHEAOaOZdIQc)m4i8TZ|-^77R19ZEwV*Xe-4#yOW_S<#sd9a|{I&_R&z>CSc!=V(B@<8I7chvcHI%VAw-Vrc{-J{y}xnDh$yFm?a9wg$3BQ^i*AlT8^ecd9%P7f>9{Wd^tG>$2T#j*_|b$IJwu-&X1{Bm%z}PNEkdBmL=QZsX6(#2>U9ZM8QIxB{!Rj?;I323pj5wsE#+-7YjHltrg){UUIFWCg3}5G!^A@D&KD0#3GB_n$ORpth+cVsAY-a@nV)Nn78pLLh;Gv4A~!gjoMy06xnW%QR*;@iRdsdb_oMMkkyc>N#Ij3AsC1GJ)We5t*P+&G@J$SMd)2YLWH5NIkD4#yBT@OSJ|z)trq8lKzB((1WE9DEnc?gATMRQ>o;Ok#V8nlYh!2qHvQ;MLbRSH;rLpC$>#&4+GyB!*a?V4_b{5anxa+H_pWu9oHy@Gg9ot4(g(qGsSFmM&`~%#C>l5ST+otW+PCO1)#w0p=~s|SnAL2v8J2s93{7bW@ZojSqeA)GmjpDb)(8{HG?XSmWkC$j+Bz~Y+;uswY>Q7A&J_Pwcw7DKc6G5r)|aL;Dfvolg~_5}_=l9D=as6YRD3N-bs%%UG-QMQB~n#+K^#KKMbXN#b7C|H8$~4@j?#T#zakBf%E6koo6Jk9m6{BQp}YEmdxx=%fDW+E_m}IE+Tjkgoo5Ln%6_67bw=RX<uNG+Ovckco%Y_^JJetM3So^KGvRAS{MP%ZrIX(}kgozE;}JSb_A!=`Vq%s|X<nfaRWFukEFE7{Ns&w-cbE9qowQg4l6j)YIGp5God8l@DBRSK%=J^tb(u958D#(1Y=9F>^S(Mizw};mrv3W*PrJEPwf#cr+J`{POPaXqyU<+uKA%OADq7-y9fkBRK0k=Fd}<fAQNIGay-<5^-gbO-#k}{91iWK1cwlJ$aGm8R9ZzL`JsX#oBX2r0#ljKG#|EM}61ezloP&0#tOAtQ)`G+aGBqb(sF*<|LFo&wWGOq=zPN8tf-J^g6r|z()mhC-3R;}-JN?KFZn+cnrcsionJZ7uUYHuK^hKnm($!Civ-jz$itIS@EUb(<C2UiIKAZ5o>?-BPP0C-ax|-U0#a4_XPe<nQ*l*m4Jze%G+GVvf7+;)SQYn1pTFFe@i@SRJrI!Y~srD2T5x#3Niq&DYQA(w!5Au2W`7+s<FC|jYzF`CCntq`rXt0}R2J{UjysHDz#VJgtGAv*0p${MO(7ZaBtER)I%->PG=#ncI=I_ha-J9iGkoQ3dLnBh?u%sSNxiPt`V(rAj4#QU_(P*kz=J%uZYM5;8I@ut~&Oqg;+PZSK&`>F1qxm?!VwtaObp=T{j(qWhZ%Y6hYYS3j#5uBC(+9f>pgrj^Y5;e&Rb!M(!qq1d*WhnOD6oeCHfWVb<$Md6Yl%w}S|nmt*tnI;%QpN?Q-AJCpHtKXU0{(!b#WC*%G}bqUfW{jK5(a3vjDX>x-EuJG?S`G3N>{pzHz8U{V`|;_Z1nM{aC%JT)DBVhOH?j-nlPZ&an0-QpG)O6;W{I)vsidO%ZD@OByNNuW;?~DdBG8)?XX29U7+?VO8=chvtq|{NXFE;{#RL%4>ww&G4$_R((oc4}g?`a|KJPvE*5D3M<9j@G8p=qD%7l+-lsFS1)C+Bu}K_Uc*n26yn~TJ^y5o7fH2vNL)~i57_-We7p<E4&fvUxK86^wZz79PSQi$TINa`!~h<_1}|<GzSZ8mEoI~K;dwT6k<<~$bbFBKT5KnWm5J<34cgRNwiQ^m&2Z*4JMK9z_~n}+AX>nLV9%WQBr=M)%U|(POM{Bh!$CBedh7HLZpP1vw{g1*Ge}`hY#)ez8j(7!W)e1CyScr7{SFA{9oq&g-8h}MZ&q*n-fmrIXWs6mY#}>d+B>=BB36le*4Tb6{~87Xwe1!aaNA|})^f5m_sPvn^_`|g94AKtmP`W@kN2E;|F&+Hz5*ubfo{mB=%dGpBeAI5Q4r1ovJNE~0rM1DNc_;RgucrXk@nh^a4-dF4j62**KW8zpu_3l+;1%;DJ9f|5u+U3`~`H`Yhq|_Q&yJJH=>R71rn%*r=Vtg<EwTTr6(7PIxf3{m3)<5d}Ksr`_f7hHwa*r{I(OrtCPc(>2N)(K<9~2D3yxJOL%BOON4SIIi05tD^H6e#t-8W=Ht878rOhZX~ILqyxQL%0BggS?l--syl0<ntFP{Cuio79RyQ`hP4ABikq)fJgsGe^IH?nip=rH0*YCV^Yi&oirfoY_h3L&C0*$(g$E}@fDP?29TRM53UM|$ICMnyFSg6z9<=e+aeHV|>8eHg2+!cm@i+6KQ$O_{7VtS>g#@(8Uc@aitUkNe#{9nJ~3{=0?QK<eLWq_Z&kN;Bwr+nMWo~3E^cwvsu>`%^DP`PRZQGH0~lvqCAC#D%|s$`b#i)E80Sqo-3_n@3AHiUZNzsq>JxqD;v=EnNkp1An4thtubMrHP}ItF4`rfxB(P!Y6>Z}%H7YS&+0-MG7YXZ^R*_7jn|NbX;Xur14^{;j;Nmbc`K{Z`ywgt$Hb7}E8Pn->!+VkPZJR?~6u=L^xjnjOwCEB`jV?1?nZoQlAev}!4Vcot*Eg<|Y)tn=M~PMVC4z5Q77PB`+eZ*5)Sr(rv5Yg=31jMCkW+U-Rhtb_PrH0*48w{JpQ3`Abx$+s+qJInZ9Wr>%?=F<&_!yx1nQL+&`9h8$Dl#|Lz5jTq{TORFNz%~^i2}y_MW#thi$7<QwXC8^Gn(FZeM=9Om#Pyt-1Wr};`lRsqTV)|0MWac-xjh3KCkY7aA^tCBFQ(u#vlI+0Lx!iw7z=cpi{LKeqz+6CKH{l90T!_gL=`^Y-3D2$X~m<di}RhLRkW9Fn@scYAbJ_7ud%*8<|=rYF2tdDo4!<O=5u=$BnkjNGT1s8Vb);CS6wUZz2-t-YO3j&J%!sB`M`9Bx{B=u=#{0FXBGmCTM+fZUt3uOcv1AYBa1Q{qfqS09%re$va}Gb*@kBaR#%4wk(^}53~wxdILXpUc*u<U`zs552^NhhAH-l<xr@uP%SlKFxcme2Y7JGS^Lldkua>dir@T8n{!<HS+L7we)>oactRLc-2V5NG?$Al^df#_B>MS${J8|7%C&FtFvCv8whYvfvqCp)RVPVsSz{kmg;Lotqq7IcxT4<m}h7q*`&NP!k53Xa``VQUbAs~X)9OhtAlIzEydvw1K6`xTq(OXP0j<N!0UJaoR&QDisqHU~y5N&MVN^XyU=;6#g4dv!PHnvZjFKl*QQ#0MbGFA`iRFm(?`!(r%dQS_Ipq83}QhJ(0^!m?xRE+_27G6VS0uA@ER`kl#xCL>Tgozy8Ay{hCQx<>d@ahnCCHm8gE|)V3mW5XRC_OyGnptRVqE)e|1YTZw)?73%AOsH~P<%t&o}hA>OekJ#ktnzQ!Pgyirs?Q{sI86^?&ph*v>4SzeZmgbC#*Th)7{044m!Bi!IB_cLDij1ccH0J5rrokh8qpwDS{U|IEtrWY0P-hka@$SO?4Cxhf#6?#sdPs2aF7CSATIVFFL|79x<PGQEyTVxx4Ip*L-jJ8Q**6+MYpG&G%lIyxb`4$U0Va@OgF-3hZZL0>Wp3Z}{}`${7a0nn+na{1@VpX*NslErPz3o{PX=0Q;*8X_yeJfCCGTF9LV0TLLP+wh~@iNDBiy#={w?FXr`V?oJL4m3SXm?|Ci!Ce<bifWpy2Dk(|NZR`WweRmCxwW2v;lvDr%Y+qhsCzt!q1ia-d#kNpADTE1h=w~M#uxzq$e15b7ld%r4!jiZP2ITd-YpdQE1A&<OJBTr*mFx<USD;LAF>3<zCl@t$uC7!Y+*`BpJ{Y&@fhZF|sW=|5gU;&s@9f-QQ5>>@vZA<u)0wkLIp<*(jz-bQ8-fVNy8g<P&t@N!6N&a7FHcZw_LhIN7VXatJ&dAty(M%hlAJ<P>g0=$Qd7%dc*NW<enQS;K#fh2CxnS!VF#QGvu0EczUBc!y!vLE!a^RT*$_B_m3<=C5Xo%#9$5pgXvF}ZMoOM`!w}^g!DG6tK;KS9`$1lCBX$G2^aoBYSJG`=B?^rjDOo9-mvN(q)v(dRHr{xa<9!ewI~Z{vEPT7d2GvJcYy`tMsWpswn!Ls^hD@k6jQjeo8iP4wA+?5dR7doln>U;tVvu7cz{g31aYk)C3<9)36$9)!brqon81e172$(*(!~ld`YY9tuE5lb-VkG)@=S<mh&TCOuxh4$fHD?GJjgzoLy1@nhUUot+E-gdo9&?fHch5T!nm=Cy8LX}cK0yIu50TRIKR`HO0*F!jN6(6EccX_B5KS{CJ;3@%NB5(a#w)VNnJZvz0|bvy8m6E(DGG(4Tt%?#tW9~W0Jk&n#7=k+1?-nGH}+7icNNHzeT$^66R=IKSXpJNi;^sFW|lY-GMen*j>iw~=tQ_=9P3fbIw=_`cp!TQ8eZAF1(fJ`_PyvhZ((LDn^|{+?t9BMLy;4ssz)u4rdKPCrkEJ`_dQuYnO9SmRg^(_iC6sc6}@D$8zeflJr1iQVKI--yDWWQvpO^RRP*3^H*VeCzPY~b-B@3}b9Z|k=c|>UkxNzVoCqhmT><&Vt?kdOZm)T(TWj7;yb5<`-P?I-b$i|W%=*n2U%CUMY~89t?dH~<_3h2|wVRj;b$k2P?bR1oDF<p}^@a6~9q;bW`dZIhssizuTid^PW8>CmOx|KeZ24|&MgNU$V8fTz*Y0ktZ}&Vq-ps^&|I(pnC@jn*BB_oq%PaXTRL{45_28OCeyhIT+WN~kudgHByg%|*H*cF1Y&GnJAm2(Vw~O#NZzGkfS}n+?y{s(GR_i#MV|D9kZy@D0woprQ==<Y6t(ui91KduylEppzhVukZzvL{BklcOohyJ=GQL7?BOGxm(_`_LKpfus~H(_QqznD8M<2T`p{aVb7mHHf2E=$UA`R#K2^_yAM$JKJ1dcR${zfmh!PfqLoC($^+f!|j@p)F1wR-NCTJCll87&8!+COBu|O(1xl@B)uM8bcW;ecN*`lyM<?P6aMx&S_6yZO%Fvc$(IBVvVMoocfqU?_{T@U<xb_Wbp^J(7%6K;S2{4&AmOS<`Ej<`FS^PZr$7g7q02{b6(uI^}_0gw{v%Mb9MXoO)sYm!D?^Q+g`tYYx|A~%*6y%{`2b14VN=cS5a`P*KulO$fmMz&rUjyyv-|g#3H9Y(Xi{SMN{}I2hNK4Wz0n_2sR7vzwO-G(o)nL4Hz__f=WR^GCv3~ZXX0(+_NBl<NpKaM4aj'
_DDNN_SOURCE_B85 = 'c-rlKYjYbplHhm!ids%|T+uGFDcSDn(JVX8)0Sr)XW0(h+jAFMMn#D%N!2d0I8~%%x!3>w@_|PJg(^~Nxpyz(I%X^q1QI|Zk;oSke~LzrM$x=l6sybQXuG)>{fH$8gTdgS-mcP%;&Z+@P4m@aT@|ZM+GNW-&FcA8vB~F~ZJnntUc7me=Id&Hm0ll?*S7};C#v$OT9vm+giS`WNrX+J^Q_6sVwFc1RUK`v;4gMHI*ndd+q%f>=-sO~Z^j1)@3!^2YVv~-|99~5s%WBSUKi_41fLdpQ(Ug{1%b(6CaX<dl_eBl`_XkXqTc4!a-G#hQ>_5x%c{P~>P585H#b%NDT<p-UCcM-EsX=m8#&|68v0>VX9e~o2RJ-Fh~U5YOA>98==ot1jmKm7a5{`eKSl92iLQqSFlIIBd9^O`Cc3DrCC|C4mszPHju2urzsl#I2!RayQMB4F&!NBSBDyLTi+mN8*==4oe^o!L!kc2Txzc5uY;_f7^G$J`ZHj88i@$uo`952&%VM)#z)a^EtiYxIaglHGdReTBW>d_~Eb|pphpttu*4s^#trk(W-QdSXvCLNubkdA@rnu~kENBKlUtHuhK*@2raMESfG(0t&?=e6uHbqvBVC@zKj+SafkE1uQKb%B0Ebw9rNW#J~HA7S$N6mJf*Vjb@EArvP%hO2HC}8J2FB=*LjOZ<m=BBd0ugh!&i;%&}ZW~28v^sjsJh(5>G@8&v@zc}jb^2m_@TP*{BAW6=k!|v5U03Vul5os$agqpCkXe46m0PO9fKT|lMif#O;T`69lU?UX6VX*n9RgJip<0xcWwc#2IV`hTK7s{weGV-*gnq`uWxmR59uW?nW>&298sMzYtIZYczYS0tj`P*(8mK96jT7eD<gjaSsD-M-&>FoL#?eQyF)q}EbtNG4FZn7VU_>x_zLo^9VT%F!pJVlEy(yN(7X*a+BLgC?FY}GX&P9=z3nl^L0!x|#6O|s|x-M4pVqL1WJ?9SPmZhNpCn#ZF4FSU%IZ)|*gL_QvOPa~DTIA*bXe>2<T2(i|TrP8@bzKNUAqF@?p0;4*y;=c2^ZJ&5iSi8ovC{@pzbG!ZdYRViJgWi7=Zv^e1q;Q~Ji%seW#Vwf`F5iwPVCBZhz2`>i32JD8A$Iy=^Fw@%dU3;Zq|3Vn=63$Wc(xZ$4zlwLdVGaX$8{sVoN+ejf&;Es&U(|str@y!GSKTFM&vaL#yxe=30Neoa;}&0_WAAVcD+q=NcBHTIz4j)pk>q`WsGx2LeO8DvNVHigyTrdFbt$7J$p2ui&_O0h~OEUIEc$Fq<TL4eTt5-mY=bS(!v1@_%k|C*UXe;o}z2W#$5ST;C$UU#)e~8b$~O@W1uKjI4s=!TB;?t;QGI)tr05+Aj}yh^rE)QdLEU_6|1GvPpoQWs6j9-xLs+MC&coPv<!@nI_##)I5>Y(&e_?6e-t$d9O6spEcvmN+fXEM4jLB6@1*5Wh(epdVQn;#6%w##cGk_p}#0<-n2Bv%eq+b*Jbu8PdC{%1(c-Atog*nbq=*~fz?-l&@>mzlsHTZr#x(|T!Cn^i#%QA*99y972MQt%&OLKQhxaP{i}B$(_c>Be|Yuw4e;H;$(tAN-oASCG5zrT^dx=${^x&Vk>ljuTlo0PVR|$<`eE|t$<g58-Sg9@&}6k9XH8bu*=>9_86Q1MqRIH#1poard_EjcCbJ}30O{mYsCogD+&npgz7PHsO&{_<fcJOg#c-G+S;Gnezj^3B4i-6V`Wz&)JYU544<4u+5<lKltDLzJZ?iGgfnTZKa7aG^3VMEJ;$CJeIDykph2lchnQs@_xM<SsIx9-tda;s?EEuXHV5M-ZwK}YC{buLc{1dRP2HV3nbH$Wt^s(ON{VmR6zh5mgko2a}%M5l8RnI@qbC4lV=#M${@V|A|0DB|MB5TH7QQb5#a-m~lK1qZH$HZICkl~zx1eqZJnaw=R1Ac&Ia(&Z5oda849dKWpt1<9T*zn`!C)jy0e`%&TuLMR3eWcYV`Z`nthoT1NLT&k5JfvfdN*dtM@eux`4V<RYPtk;TS^?~zDpWA1c#sVekbrMc2A>Y#vljHG15yJ<AXg8D$jPn%a+%xF)0VBFe_&W+oc%wjAm(}rAJZ#1mYQh^V~cU_XOr3ZQ+^BMRO>8o!!xk(Z69O$a|Qebzz+{XKhnzJKSQ-*c#@SsB8wE*8L%0-ZnQSQ_RoOmX3qM!s04{5gvlugg#IGp-irgIdr{-L3~RI)6MqC;(LfTEOXgMCOwR_z;`3lO3<1PV@p(v+>1Wu^3{5o{FMx-yG8osUBK!`AJLuUNsln2Dm2UFSn;50O9Ee#TjAz6VnT~<16$4=qgTD^Ozk>J>5561!csYo^i>R_QR)(iL!X--QSKHO6CSDP1N+SA7D7CywAJ}JfAfHh^TqV0qh!fj#82zie8X0H;(J$EdNnKZUJg{}6<+j;G=O_;~z=f~#@qkf=V}><`6A;D?T<|iFC)91c8YX^+f}PzWg>%a!0#+33CJgLxtEs{I7IbJ7sh7yanwaFvV=F?OR*P+!6GDvsr%+FPa=VO8>oEu`c^Nb2O)1WVppHd1Lo)(8!dak98<a-*=wBf}R?9;Lu!zNi#RQ-|5^Iimt<^b>ZBujHz}bys=Cy+p1w<x1f0M|5b{CR6ghh;=^sV7@(B|?C7H@M^=OF8%0_2Fptj5(qx<w7@h+0>WZOr&#4=%IM@gc8g4J)vSnP>xZ9Fl@-hNqd84-Zyc7A4M<M$r+sszEZHit`olpMeCILOGv$28XH}&}Wz1jqU}ijYUZ;$K|aNchi6<7mF5W$8wM}&7_pl5^f?}9IGaZ0CmembbO%Y50vpzUBjR=cDhW91+W=G9S4e02FPoBu4NiYU$!k14QMJKsB2IbU9kWZd^iFWoapAN0EO^Vo|Cp9@J8bC7!@8l6u=Erh#?DkRK%0<1nAaO20jEb<*<*8Sr#X|gu#trO0Z#(>m1HB^%@Dvm1QLwh5QQG`QfP<U<oI?X^-3MXus`sqwE@}yciPMW2ZrvW<oy^B7pu6^U>i^>-f>+W`;MOSL@r@Y%yFOm@$$$YKSP9!x+5@J+j?BMzP*+ja&VQd9vFwQFVrU?oCU8szp_U5CSCC-k7$NgTQoEJKT)e|4egY;N>8a!s(DA9|tADhzMH7Wcym`i7&3Wd3?|(Yz)<A!k%uAy{1tAx4|l%XHDJ=XqOSn2-XnrHGsndhK0ZV_Vp{T_T{Q#9Q;PI2vj=7e-KUhx6H5evY`?(L!^M+PxzH>S16I?p9f#f*$Z;ecDXVF&X~UoQLwlWbyV-Jpor#>VkI;H7C#snn@S^>r*+1_K>f>c12ogYyvpl&J{TIIN_Nel#`$u+x!uvL!tt3w+D!L_$a303m25Ro>`qBvV=RNMkhnFQ&BCt;Me$-$T}&sf+GLXqK*rKGY}g{Lszs<?_PJFLHkqvV)zjo-LAi-Wjk7sa**DR-Ptb~{EPc|7kH!q(YTrf!C)>#dTNzVp!H<<WMeJYvz{E5M)`JHTQ7iHa(5`iZ==+FI4Is_>vaHVIfwD=G<!8L!45SpGR8f|{_#CB<W=MW#Gqd4vMQ|(Gu7$E)dqlp@ZV_hyf#p;Dg02!~Ex=^6by`+)VuG-f*4x3*-z0E+ke*6_#+RUg3`jUtA}Fwi-@Xn>CTxlo2+yE?#HuzNalo#IQI7Vr-@ZDKwKhON7_7$qMV{emBNKuhg37NuDTA^xtM;)H20eXmiN;5mBTdy#t6RrvaRntU@bY+P@JkR!-c*~HD2(z^?C!ZYm2nBF`9$98!*^&K!T&+z|E@tzEI>L2s<ZiU`_ixy5Q%Q5WwtzDWT>bukLh2~)rV(?Gj&dp4UjC;sa(1MWi$1;hm0aaMet@{poG=GW0XYPMS`|L4<Q-?L`u?hb>>o_k#SkB!R=zO2B#w#TY;Et8GGAg347yY$(IyxoR$k&2IwnRUIDOHQLy%Fto<5oBB$SYe=m!sVZr_kmX)?U{zc7JPY!4ji;yl0tw_x@wW{_$t-c49750LX7(EYVRA%!}O(ZdjJ%E_cG^t=Z<j2(#0Yyth*>x@{!Pv)?K*Y0TjoCAKi{I2Rd}$}Ksz_%I&vbVW(;QPNPQ>nL@APNab&B397(N`WuF2a!eQ8R_mvX4eiamWxiKd6a>WZ|F9fWu_B_^Pz{3;H1`_c0K17;|KJM3CBw)bl@hqnF;p5d>iw=%QigFBx`+{@6?1vK-zi61@cWejLwi(7E0+f=N9*PPdCCR?CmHFjArmM3~f-$o45?ru2dGY~JfAm={3>@zR3rZIiJMh*7^tWIw9^Y@^SL|>-Lkv{#BUXUwS`RM&Xh#wC}>_sO}HjW&$Fw)NzAzegiitz$z8aH`)kwnfhl&E>KzpE<Ya6CrJ^)@zvV%TY2Rd^cNvl9ub1!ahbs%kPYFhJJNu#f~@JZ2=!?@588-^VBv51g*Mnvj~rChX6^{nt&Z{8MA7Iq)s#&Hx<qHU8TE#T8ffl0|44Pu0<)R>@f$5Qf&|;5+`0RRnCG{~fe|fPmr*H?iY$#YX!LS9q)&;UZ~AqYKGv&|GC}RE9UmT1c5EwT11Z^brp~2lg3gT2-B0<|#FnMtNmZv5<$+zhE>7T8>Q_NK0sH@d}S-SoJ7`bZLaq_>0bP@e>uEl0-)ddB8!0x`f44GM)vW483^^iXybQzmg4!O&-e}l&`hB%t-YXp>6kQLM&@Ks6no#?x%JnLdukA8=$9{<F4{S3*|0W=qIS}TD-h9lJHs{Tx|m*DIH`s7k--H|3oh>S9WZC-9`c(?mI*%?Hi6HA*f0OB*^5+Y!`)KuK?i^)^|~Xwn$qao$C>jtq%4uaad@Za~9$YpfFuk6hyPvN??QDrEnnLaV@+aN>KdP**tHvoymBTv<g{UYcCdSfmZO2o{n1<62b?$o5jWr)bVGNbnqv+us9)qWe<0*ETN%yY56#8ukHWDNRGVT+?s=teFHQa*Ma0ak?a5`vmR(jRCj{*q#K%vAEoIS&#Zy&8}?m0s*UDI+Utk*=HaYu5Z^VDFDYBI_8Np);xqabo{rX-Q)T=_1fsD%))wbuVQ~ICN9k26(uUD39B||+?DKt41!XU%&&D9nzwC04auiq!*fG()oqke<1vGrmuc|_BR`yCMYY93#r_?<NW;q~do=?xKs^kGiKgrq*c?gS(3m1!pQ_3LW&|;d03um|z&=p!~n;ZuuP~}Kc@|20PYCE8mH#F#+3Lr_WQTLMW^oDkQo*BXUZ&``vu!W6VWt6}ua~l=tgad+_pOxiplx<vz+c>!FR9)_F%Uk<sT<5gY;?bewWU@_$eq2P`6K}i4P)RhW8$R|0-QgP@OCu0r4VRnURkGbxZu_fn2!*q+$t;Vc)Un^)CSTgMHH#7c4uuG7enG65Kn@W;0<}?>Ee+i%!B92M$*%#LX6%jG28Fn1k7h<N$`*gB(W%R0=grA%HbDGC-@<Eo5E6xk=v7l|tSzps^A=CqZCS>#VjYLTO8A_&s62DAF&l<L_`=7updmd3o-D{!XL`sp!H6_GLHjr;7h<-1QosW9Y}5WJ$e7NoMH|L5Ja)fEkI_Ng9%2#%&Ug&|lz}QZ4ail!$m2L{Vi~l1%}O2ZTn27?cAP|$nb{$<@w^Euno~3mUnhwn+F1w4!#jdJ>;ie~E++S&IFvg4>@vbsp^rIiz7=jjMcs))tymJT;%489*}fHTgB6`vABu1gK^^KKeW=4z0KUfodYAa29Vg~ahradXYng6O3Bw?2+E~;C(KBt8*vgjMPhFhZDgj(gyZEqe1rwYi12;?m@TNp_(E%k25&ZLQVwugg9=kMn^+%bOK_@CyybxP~KZ0VUrD)!_p<mY34WJL<q&InSd9^X&MuPC1W>NPy9d&_Gfa$V9W7Eqq{N-`lAtsS#B<|2t8jZ1IQ&gY%F?#d}O+TRVO_jvOKmv^vUv4$v$@UD4VI>Y^?ZDV%C3)xc+#ai=xP?Is#xXrPfno?Fu^<vrjn|mH4Hd9ux*Zem^z7)XLA736Dj2`j#fqJPgkY;sv^nhY@!#ug@qvCy{s5Ih{iZHI=or`Er%rS`QRqltt^phdAOSN)b&HEHVcJCwyJub&YbUyH$7Zn37mfs0ftz<x8aT2`&kA()?ZpKeV?i_&ei0&rSP&kK39sFaC!&xDsa2}CgI`KFo}l=kC&CwHRUAtq!U)8QN!gi8dGhe@838+X;(S}pt!%;9Le0IeCb6>AU5+0^wMm;g6FbCiu|2_MIA}|EU(yY6G{aWVA(PTAsQ63E|HXoIcZ*H1g_(U%ON*g!r_85%dS*T=+nUCPcZ}ixRf^jpMwbf)hXD%`-dl4^!eb!Wb_2>sgX#$FGPY@mZRaan+Z1_i5?#|@G&du?#~Ry-!D7<Fjw?NjV?oavgunRA?O^6`fBhfsr7#yp>gzpa!$U~ibx1l7hHi5ynm2hGMRS3EygbH+VaF6g2dU#74o+DT-c+j*w{vgo^5tO9?%i4?&kjvT!kGT>DCpi0M%l!}P|fX%KBTdsC{NYVQT%w@s~IEPC29jQtCwlqj><=u^=JBZ%BXZ$_AoezW|ObG2B62e#>lR?&jwHE>X_}forc|wG!wC6E6co*jXAU??A3Od=D>J7HEscz)*|0zAXC60wBE);lu|TguPvV|rm@*6`$OO}$Ura~>@E;!$^fEO0b>^S;TWxLDR(Sl#8G2^6LCxM8-{*ZM1D`PVRB?-j1d7y8b_ubBXMU#gBsJ2ao5-ItCAL##-VjtslfKhj6%&KH`o2N%U!~kcbBPGBD<H|)gwZ9Gqhq*LlN3)ebJ5Pt{j)hC4rK!j#pt44qp+cqe&@1VS&EwQr(TpS7fb}m%vc=`qF#5YNueLgfIX;3yKRn<&=E@+)ZrRYe@S^G5dge5GmzBC6(VRtvsN_@(@x>+X%Y1^l~=|MqEbHGK^zIb`M3|Pb1!3{-KTA{*3qh?CvL`gkf35?4GBigl5^y?4Bp3<xBoCsn4`e`R%a^?coE(I1+RiY+@sScwl3UbOzz&t7DR2>i1&GfP6sD*J&=sZ}4U_Ml2-3W5EmZS63l|VP*(;tSYEDGh-IjEo6HW*bSdzx^$C%x4^I<Bn<*4teku&kIseyxs3(-y$LJTRR%i$1DwKv+Vq30;|Fr9X1Z#+Snw5v19L~rYv$fs$9=`j3g-YT28RkwRO~Fw>#~UlXH-5@cg8^-{)Dy-4vz9WY2B>$D;zkAg5Ng{&EO#rL&w=l#`SVj=Xq>u2GGB6_^?6cJ`Ieq$~DUzSZ*3fFd{Dr=y*mMcW9FdLFIvv;93D;2bM*WX#z&kwy5du5=-yx*h<H*%y(%2WorD9bTPR%CjA5A;O7s&C??^>(dffsL?Nd-Y?134jnN5|j_{6!OXE8(K@8>9W?A9PULly#tduKaWy6o|7H@y*PY;7<=i3_T1i1?wtu>`ovL1HkY-ApCJr6nzu6-Sx$VV@RJ~3(aLU*PU!GG+PW<ej6dBzL_Wdr?L8VY5F{tLxzn2F50KsBdg(4p(Mg0#h5&#9RS*Ou2nCB=J^x5i|zv+a(4q~yT4u4Gp2&h8_B-TK{0E*8XkMvtSv98M+^<7lEj7vmQ&|CeYC=)Cb@?#J5o<cpxk*#1La7kLxA>)E;psgelwLFjBO5!PzuVzYIr5hZZqn=+bKoU6b(om<kpi->t9B1Q^&ol@o*TNp~W;^0=?r?j?AR~pDa!zyBEAl@Zf6&HE4;aKzJVQ!zCs(ZmM$MV+(Q$ng#if(GitT7k7kfj1JI>_AS#B7~rmfa1G`QC}kkNpAYzL(h=@ApHaLHs4zB+m~MOvd>9bU6Ac-X_<>fj@GkQqZoT=ox5hf(+^W@uCd7+v`{#gwW23(h(SBf^~ezGRL4AT8jDXP*o3J<Z+psxv;wr`%`z=Ja(s6zLyt@dvZE?fZd>G*UP)*?k3*{jNbVIN;d$pnQ^Ghj<b1<TAn-qCQOXVWHJ!BB~ADpTe?m^oB$`RR602lCq^K|6xSH5OC~L-;S<{W>b5E74Xc}K9X)oy^+37S`7ZnF@reOEhWE#PWjcbTuIBHArt9^<`|9v7VR?Y_kz{Pvkj5xVV$k3KZ5M-p3b3kDXl)dkjm<pfYdPIlyQCP_vZEL)v@uk$L;3k4TUu@#Fm+q;sFo5&nP)YTD~9BWT5MWK7x_H9mGx!0BUm(;lXF|P23FRo`Dy$f{?7B)@87Q8VG_%18@HWZi8G7oz`a^}k)e#}%%BF!1F$UG(a&NYm>UvpeL_B}Q(efcS)QFs<Tnm@DhVKIC>;0JXw>%H8>5!AXYCdEV<`q>=#HK|efEPQ&~rF0-NZgTVBo$N$_ztQID@CQi8dSb0Z~~a3aj-REmK)t-a1+4t_tM+x8YoIHagY8jah2dip{~cGL6d7Pg+6=RS~6HAy+e~Hx%cI7>7TY&Ez5_jfnEB9ax3iTJX;dv>9icYNp>UoO`2GaGjO%tGx#{kbZLfiRIAmjjReqoRb#H5W83at38#M$-ax?wppl^P|IU+?wQg4ty2Wl@$E!HZ8v{M#eVr(SEUp;^Smg%BMlZ$Y@0x6f(MKAYOyC0&&xlwaS%+$CJy3I=xyl0W0Iy_B9v0t=afiSZw~9fBdn*Fb+uh%dcU(lS<8!aVT#rfmI|zJ=h?he!RR1|Cr|nQY}m0l4e&tAWTuUX1nO1G_EF8e$i9F+8Vm`V?oCy_O5JWsbAkh{V-XL+sq|tAr$-v#v~DHGy(^i#>-Y61;@R6}!?rO#30*2|uBvTm^lRi^E^R1AJ&g`s%Fv`Q45h}WNJ*Lv?N2pBZ?&CHgS#aVQzH_nnY6=azHbk?IbKM%I^uF&SJ$-B?BSuTO5VI(cY||)^|(s++C%}{#?<>BxGvXCPp*eFXpRs*z;@fc=<#=vSKHBJ<t&O&@N)%fABz{)pmS`bY9Uc6SNX^~?l98OMC~*?Z&FYQhsnMTY7lU-LG&nswy`JA_>&|L*4nzR{OX)(M~Bk|=(0VL7{&HJm~E#TY^Y9*M*rH9$wooM9`w5D^pM+bxh{z>FqUvzEeet^4nMc+1?5AhLMY8ius2&Ne3zqR1Gfy?!5Y5>OO?8sQB=UdI*5|)+7>ZNN4EvTUCjg^P^vdCNIxkD(vyUVp&j_tTA|0&U5A1_f0QgAyz<jV=Ro453mw{gBPXv#Aw!Fu&16uA-&S~Wy8*};5FdbW+=c+Omo7}n-MhWGkcU^nUJX1HnG<0UPPS^VaQ|2&O7<O)MAf$$mBjPDqmuZ4z)`t!xiT`TU;!nR?`*ZoKW|K{T^d0E+TB>jKyFxVkrqXvCV82yF;t|<=a#hFs;&;EP#2gTIA1&RTHIUaEUU{s#7%vA-BS#035S&A8y`{t0vWXF$kNc@yYbT;{pjtz>cu1nPc3aAm_<f=Ctfl|um}?rc{<kc7g|2}@I8DkYp8OR2Q$V5w+g`zp9ntd4A;_FNxF~To<1b$g=<&^O<Tx1300`r$v)fnqTDuD)3)V$H;MivDF;};PdWOL9_MtFjr_@U(VTA!bKYl=p&W)@s3JDQo9k-ZEdELhVTg&c-|(u474ihk^E%HynV=IPv0*X<ofgRxtpb4^AxHe1iuVOaZ5hEx`sQ!4hJ?F?alJe4Q6)X>;Zd{&TH+voqT@#l`VWOkbuZ7XX2BRfVvIs)j|v-a*f!+5Ok`Dn6>fSDziNj8>F%fqIZ}IYu5!~ps3Ud%0}nG%t(~g49-4NBA-MMwq6f}C9dMj->=gPfK%Id3UJ?_)G<Vyu)HgVJ{{Hp<Nk4pi`!4<Y?Hj<z-%no9(c`iPK6Or?|8Mn{&-;_-KmQxlJm}$p0}cohsNsb8ty=md*voJDK>o}B9<Ers;$q~UjXSWit{GDHxywcwN+l!P?>191BT5%JnGty81Ilt@%))nhJFUmhkAU0R2=QUM!TnwE`)A$!UBM2nd-YGzOB#Efn`aB*vwBMrA3eb-Zt+<^g&BG*PI=Qg|DqBy|75Yugf+(OEm}tO9s#KHCDe_s&=+29H{;zdS`KQwr=yk|@8_-+9sRLRTkiPxI58Q*om`neJbU`&=ukG_8DYoO_Vh@qNzp~uySNFdol*kG1!!WotXq)8euoKy3h?Bj<^St;{B;TG#HC$q1U!Dd!De@3*KJ>(-~C-YdQQ}2r^`+V51zM<l5X9mWo=BGVJYg20gs|XVd=yUNqMIS8QhKM==R5+I|#~H*QXqvY6SQ6?1@d><=^v$7WDGTE%|MHecm$vck=idZu0-O-|vsJxj42JvAeodT!)c%W`GJ?*!MCE{|?jegP4aO#6%oa4;h5v5?#&qx8I3#Dgf+jkE^KDkR^{WeOdv!*^<>~u4>sQlD0)E>YMux^G>Jh$iLZ=goGkDB&;o={&o=d0`c`9Pm*Q0j!m{lr}ea$km8b+^eCtfCei|@mbnALn%`Qie;8rCWsGr{N81h(xK4F{D^SSv;dCJ5w;t5u@!<5azmoM|t<1JgHP;5PD;i-S4q$DT2}gLj3L)xO>Rkc3=~>|6j6idIVGE`4*xXnOhU5kx?xoCs=^V83+<nZTb*=7O<RJSSKUZ%bgSW5Mf15mH>IQMDJ4|nY`C<b6JyH+!XOG%s+jgPiSx%m&hsipi0w^el2bb#@Bb9eRrZyArHqhFh{Vh34KF-F`hwU1l3U7=Q*hDwudb}AslZ|e!@>Qg6xEHJSc7r}9<gX?Nt^3%Mo5{!`MF?E`T*N={GH==(RT6x6rplJx!HpaU3_(KHtLhIhYg9=rGQ|6^##EfR(z%L0HcU!|oV41yX;D@>sF(An=<(=>Vu#{Fvd)9+q_wlkKV}tXw;@+xvf|uH<cyX?#fIE)R*0T75~`15HzNvO6|mEOq~^3|ZV8onu0OTJMVZ-&DYu6O8C-}^GBC9%i|ZHxCXvpZOE1i#z{_#>^opf-sbA#crNN8@;MR<ZIar#<Jh|;SEI&6u((=<-%ZB1Koq6v&2=p#tf8KO)^oANm#$7kPZaDh}BFP`onL*d=kIjnVrD6$0xAEpOd1QCyz!<jQt+b64G_E@tj<1XSCbql7(%ez&s=K;h#?O>e&buSRF`sP=17H43)q9og?3(RUi*swraxZO7D=Dj!5Ja$LnVlp2bc9SfY#zX+vaS}a&@PG##eneLLEHUEK@?SiE2>A3x68QInv-f$UR^Z19Sxmc^rib&7Eaj!ZFBkP$z;-%{!;;H<^Cif;pCs4eLAT>JFuXjj2xh%nS+rQq|VOT$WR?QEnIB0vr%C%Yk9;r3wB92|EzW15ggg~7Wp!>lxjykigtEJ-`jD@c9<4bbVd)b?m-0}1ko))=*OCDoV6Pxt)Dk77{hS;&lE7smFYQK1u6TyFQBh7QFPN>EUmCh96~)@;bcSEL%!Yl+*?=LEt&QTA9vA-7o3M)=+{&w^2V2YFu_mwNgr&d-;u-%OBi$i`oqaP`Z8W;sKl+a<<nT?!{qm4Od`8tvFl-(O?m?wd3BR+CZg5Z0L7j3Dl0GS?~auWcsJQ~MQteZs(fiz`C_UJmG=%YL#wBb2IX(T<SOM&cH2brY)#Kak_%9I2-U+fF~+z@6AkJdHJm)cS5vY}THkT>@d~Jiin*?RB?LfXlsi8t#6BLZqHN)R0u>Ed18W3V`0#_y#DR7ht-pahM_*T<^78<VhDW<`EV8L+_#FvP^v&%Sx@LhiCX}ewb>eZB*)D8~yO4DGjsM65-^=r=neM54z^RpZ!6<rm(-1O^*ocHZwX}cGBaIEi>Fa5tm_2{A<3$1I=-md?Ct+pWYrAOiLoI!tsz+@fAmLKzRrLF$UUF_#-Y55x&6gBNha~cd?Gb?1ewoic#77&1LeVTBod?O!fTZj|sb;6bfGjM=kyLmd;_fy9;#=bnk)ddh`{&_zyPV`aHgFc%gL+ZNdnVe9nLnHzcQ@4U#u!^vN7^o|uvgYDC64RoA$`F7s3g?@D$cVdp@3N=hpqLrbOPJDa$jLPL5cV-uozOuOOj5N?W!zRpJK;<CkNL7!>QuHSYU`USzG|;$QSkVVX|9w&gZ0iqalOL<YX`4PsWLkK6+l+AwNA%KdU#9v9DQ2WCNmR=AB{;h^{kp*A($!jt~05%HL#^=#6U-gKLK^s4luL9?Mg!?|!m39>205gAh)iZpPBxza%3+dAw>KNT(hI6?orGp1a2UX!fjQ8z5+Xkt}Y8P7){+ZIRWIWk^0Y+0DG8-ZLFhQ{~YNj{prfW{?M-pKhO#i$yg?$Xyofowq@qU<ilt?s|v_5I997dO2MQq0>GyRF6?9bdPiwE$<!MAnlocrlt^<UA<Zv%6MWbmF1W%(?f4Dcaecu@&k{(8q=s_xCuQsVg-rnrr8zfrS~~Ku$4rYWpxg;Vwk3VX-vLU=4WiLZv5xeF$Hnkw_YqhD{q?YCv<~H=bqvX-S=GQKJG&mi*TKn=#3;<ovwhynqT-M3U@?%G@q0EZ=Q*m2xp6P4pRHb#9`<mQj>!-0L>+po#5Oc2RGwsgKMa0Mdv&X<g2Z2&Y;aq-iU>^{OFo@85U_u8#v*!c<6!3QvBUHniFbb>){!zlhh;L8R70x?a*2)2`Hb$prwKofEc0}4Z0zMSVH*2L>{G-!%#<744n+?n&r24vn-OL%npZXuhUctYLwC#KLN)|X^$0il7C&)NwKKgryr2^MA0{J@3$f$y3@0XfsoCCV>Rv?qoH-iP@u*t)yhHE<c)}WH_PwcnqXft9qHlF5;{yUfdVM`Wtp=Imm23*tC17!VI1+_(&?_~;+y)~3%u?Dil~jfuXCU!SZ>Y}6>UiwMWf9{U3~$a2{qDh<BqSh8+@_FI$vTaabNoCaAuoVsp#3P;&NG`H8H!<)(9IipF(Nud0vHo{WSb?-42syU_kpnRp+?D@Fk_~sz3=Yo{T5Yl4vsi%QN-rF13`xKW&23;cyrb)_54%je3E*x)ho0QA<4>Pe2qAj;?MX61mFR4S=<{uAG+o4S83((?VCnQkTkVOlvZ~s;Wl6P^z>&`9!iw15RIq?;|UPUsoKOJAiJfy5XoEB@Vtm&3cdX&~Zo^u`OVc{<+Pa2P5g=lCPaBTcv`Ck=ThlvSHw9WNAcO<ynmZRb;^C<64^HgE&z`eE0nH=^4%J5)=dNEl@{3rp%;U!I<+BYih5-=x{ib4g<AstX3goF)Z!u5&se#t3&!ibCfS(bomaqzT<G)Rd=4_S%Y0rn&wn}PdAg;3Akvi0IFA;R6~G=uSzvBoi}2~u>1i%4~Lx|OCZKy6Ey@^)K<fA=n?vREPUY&ns9$T{*_*a#jt_(IRCtkN0ae0_=Y{7;VLH4F)k!abqp4>>_5^z<6@az!a+>(h@MPZ7N7DMd-N$A^pg3f_zZtaY}1+XD}~sE;@=?6;*;Vjn{hs;%|Evprggcdbk+BN-LZqo=If>?#X}SzZCq4BS>mXM;TY<1><pv6a$i`C11{vDcyLj=h8Ov&LOt7u(R!s(H5$)LR7Y_Ks=>P0HUs)zg4XEVqBugw^Z&P=e^cwZ!g7aUSN*z|Z5P-z9`njT>%g^;Qm)l>8g~75L94Fl%6)1fXo&56ghBNcc!oj!5`Kt5^xg_g0oJi#<)y9J$f@o90>VAK8_0Og1wqn}TXzPJU3VXrjva9&jmb(B2ypwa_ItQ*Gpw|)0_~W<Y7Aj&WNTN3t1$1MTGfSPa>Zk+csPeb7M!8$(pB=EiXo}$4ZcS|1_=;Kr52|IT{9tdlLF*PUJ3n*z&1}E*+E%&{Qi`cUj?kFn}^C)CIXqPos7g5lhDttVp92;FkVy9ByTl!!bWv5uYD})Yh@F5r@Oa3-G5qhP4SNyJ;9&&QS-fQ-`t?%$xek=_hEH{Evs%-gy33QJ+1xyi~Yg8<)mT+rmaWToqo;B=qyq;O%9X7{$vIvx-_CX6Xp8K4Ux5aQ0TOca;9s|;1Tiqu_nU2E;Daj{L0UBr23^b*#T@AU0-%?yeB~Lc}9AKj$3s`=V>KL<aI8zsdTkQms&v0(qgM1L$mT)H8BCt=03SEZ=ZRc&AOtm)p`j)9nYefH9xahnV}4PocJ@FwI70IWjCbihnB^%)>HY4gLeb_&y$IH55pP(B=haNxBd?hqq7^pi+d?uvs3SKx_c=dwEsJ#3;($SsV9HF0*w*W&&${Ed((w~#L$f?TAXEqU_C5#i&7ju`6fddHZ+I+q~CNMNIm#0f}7*kVwe|bI5BrPR33rxBE!GpRWT5&wdIU58=20O+7{|f+Ag5|Ww1w+%S5Dv457-AZ0>%Y-UwRibU=@?-qtF|7LUK+-@Lu*JDXjGp#dfB9l+|VyRU}(T~7~T!Mmq^w>OzxD_F=7*zclyAP=apJcOK`hKGG~ucyW>KPEyodVHRN2P7PCr)$X;3B2)!8f)itufJb^7n-$lgC;(2*FU&jm<>rGu%gp%^>DPdYAYHHzrEP?K=O{qkrhif?;#%=PVG{?di%H|<*;zziYpB(wNJb`Z?5nfMOE6P6M8R}+hzNjh~=_u;V~T8dE(3LTujoGQaUOOG={yg$%Z?U6StdUcs4f$;YQuL(YcCg1^aG}h~&e1@!<#%jidjJB+2Lm!bDdii<)6OcZfrJHyGj?>)sMc`iZR`fWjwE2nb3hWJOJ!11T}>yrk34+2oH$oLIbaR2?-UYWCQ_J+tO#K}^mP<JRpU3EFxL)JgdlSWle=a5yyEnd2Sqn2vC-><va3(pen#&Cnh?g9CXSIx_%R^M=R)fLP)j0G!4WKPgPZm&1$izI%`ymR*xbdh*7Go_X!7FP7bs_kI9Cu^*#m@6*`tOka72?3Dqk>`Q3b+6+HUcknEe>St$)%l>rWZDUz5(<PeB0<C2-OFhON!9$Lid9eHZz?=&^ge$|YI-;5o1X2$*H?-tcmd1U2k;`-Kkdr(1WYg1Om&#}?Ii_C&qeCBd+%BfMP}g?>5dJr^-e?Wf%D%FpG}D0Yr9%^2YCE)lBxG<=j2zwhtt<MGFs?gMlxX@CK0dolXXfYGaMqHWoz=0LHrlUTVmWhqp0)2XsNNh9YTMbdxIl(&C>jGU9FtAL%zPAuaH^Gm^Z~GuLOM-DQx--<eNkrG@9y->L1aW5dIY~GsL+4}Jl$-ca6sE>;><=reFc&&S;zM=e1`3n?oPRb$udl7cePvM43w=hW$XWaEtM@?Xk(@6VrX#IV|Z+Vacz&(x_9Lq_=#`nG^D&cpfeEj-oc&h#cH}h*tIwW486#{wV8i72t@|&dK^KG2!{7b1#ERjf+`P)e!G=JWL#su8cAOq>1p^C1ZP5SXlQu0Auu7Vxv+Zh8Um;|Rwp?1dk+M=w9Cu}mU=f$z4ads`ytgb3z)uWt3mp-L?H%2fOPGWqAjxdd<(MHtpOU8IK`lgW)3Gu?#(~xAWYbSo2r~1=A$11FVO^_4d5rfadRrN38#ohKPlp{EWhn|G|TrcM>Dd#^We<ALOtst4iBuphujaAYxRLOeAgI@w2`PfN9!V#*nnwIPUwcJ&H&9_gQ$?)^$vIlgLcso1ic#<EH|?p;xjEH)bxx@Y8d9t3sDwcRwe$cm~!T^Vqsk^g5%2FbG#?EJe1R(0=snzGvW5%BTLzEuV82gW;<d)2S2E2@;^Mb**I<W63KJctE*RCRW&kdS6cQxSKh2TYMb+d7kqG0=lPdBQWtDF)&-SeZGsdry=><igwpZx+~`8R9<GvZ+9~<IhSSc;H8U&dXDs_BfzmDQo8D&Vj8XXm;;Mgs{p#BUN1tztl7iJ6HU#`+!@r)_ms@-??H&Ditg|6B<TC826$_UpZyn6g>>h=K6XzZzJ_0|Ixe|%wjye5hq7pj?v$rMDz@t1MQw}E|Y4Y6_h&~<!Z!hvZ$Jqh|DK)`(FdBhA(2Ov~XkbE+7sBJz(draxuTUwO4j#it(#H#SqU{R;cF<)GZ8^53nVzYjxI|qLPHa3`LOT%Cg{q*NDSht+&zof+n$ULX9}-@jlC#lmk0wVy^tZ&N=x+b?sJFF!k6>h8N>E@p0iFRg86Qph2U2b8c|Mx2kX|qr(ZRrKm6F3lq_jw{kNT0qF@6DP`tC1s#@tN}N*6@<8H4M?fz8n`8L-X;JMQb8FYAnG$BZa@WeC062GjwBoxdXx0!Ai@WpI_B{jl#cJrqbC5HO;Iz~ME=L_xhkI2vL^(GZ<<GueS65aC8BeJW`8gab0vNToRqp#$UK`r8+)9qsp7p?pOCkv>Sz`q82MI(-m+>>o;5TPXY_{5{42K+HSzBYuazb77wCMC_i#)=#nG;ZYb{IRYK-AwYow-Vf@qAKf|;=Lnt99Z;T)C;RSi3M1%dIEPOTCx7ne>1B1fn+kDKiZ}M8bnWGkHq_t&28}A<E@pf9!)^`$Lmz2fAOyos`(;?R!MF<0f0`YWSJiv_+Cexjh~FUs7>y<<0PHn}9!hz#i*sri`)u+j`RHgbVEb~7XZfgqhMKC?!5L<Shx-kR1#Kesur%=O*^?jk!#8>$N2pKkaqvv;3R1KGuJybtyq%)qU+xSN892qGbo1S(cZG?Zd<V=Q_oHOZLAynPJ}|3l1X~)~syB58l=?eGV`RU;OSe5nS{b=tYo)>5ud!8o?%JLf08RqVfY{Ss>GurA++!?OtS@7zj=FaUDjqs$9F=-ZAF$&_U&H3>KzQX!1$u*4NjF!9cn=Sh*e$mNCRvr&xwUCvC>z~yxx>w+_gj2=3%guYIftlIYh%s8e#K2>iYvbWt%Sbu0G%5!Sy5s}z&=-!G@?ug(WB@uM~-2IZD^6Qq3MyhmNlor>Dw14uaEgXGCyy6B$X+&6Y<jyARTCBgCC4)IAMT0kbi%7^8S}sJc`yU5PywiIss05$SKM3b#@oh%eU|U@%;UZ=;W8@umAn|M*?W16YZ^~5g3z(3*Z&LcKG4sSl@?t>~D{a8N}K#WDueY>cz<~uYNv3v_}6KJwJUHeSH7?)tgsu{;n|{>uc0#qU_Jk<|Jkt1{c<N4o@kcV^w3$!zSO*MIX8v%o#f9BUT}pr_#T>sc*e-Om(AaiZ|{Ssrnitv6}1Yp@5nHSXl#Cr0q-{CMH16DaJk97KocuFOKknhf4++9DHmoFJx9}XGGzLF*xtDaUdg*wgb7N2W`^n=%6bJ;vIYO3`U{cF72zLU47)A=lOaQozNe8T@fEvoO}7xnRj4*AL(4BXSy$QWiq$J#9wrJ+B#p?F#sHX4M8-%WA!~8O!e~ltJfzlW@-}zt2$}zD59%cDGz4=J;-yFb3Ml=4)jG!b-^<3xbO<wd63>z^(Q#&Z&Gf`moU52w1^HKC=W{)&fQw4DPQi#iu><u|3w$IO6X5gAG@Br(c5@{ND7@D|9JcUf4zMD_8-wpU}+!l(O2i^_b2b(zW-=V^#*G$w+2f1_xC3sj#UgDJ-(+#*2U!xfHu-}h331I7-gEG5Rj(ItH2V$!T$x>i={O'


def _load_embedded_module(name: str, payload: str) -> types.ModuleType:
    source = zlib.decompress(base64.b85decode(payload.encode("ascii"))).decode("utf-8")
    module = types.ModuleType(name)
    module.__file__ = str(Path(__file__).resolve()) + "::" + name
    module.__package__ = None
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")




def _extract_run_ratio(run_dir: Path) -> float:
    """Parse the formal training ratio from names such as r0p5625.

    Examples
    --------
    M2_amp4_r0p5625 -> 0.5625
    M3_amp4_r0p21875 -> 0.21875

    A directory without an ``_r...`` suffix receives negative infinity and is
    selected only when no ratio-tagged directory exists.
    """
    matches = re.findall(r"(?:^|_)r([0-9]+(?:p[0-9]+)?)", run_dir.name)
    if not matches:
        return float("-inf")
    return float(matches[-1].replace("p", "."))


def _find_run_dir_highest_ratio(runs_root: Path, M: int) -> Path:
    """Select the largest formal run ratio when one M has several folders."""
    root = Path(runs_root).expanduser().resolve()
    candidates = [
        path for path in root.glob("M%d_*" % int(M))
        if path.is_dir()
        and (path / "dataset" / "seen_combinations.csv").exists()
        and (path / "dataset" / "unseen_combinations.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            "No valid run directory found for M=%d under %s" % (int(M), root)
        )

    ratio_candidates = [path for path in candidates if math.isfinite(_extract_run_ratio(path))]
    pool = ratio_candidates if ratio_candidates else candidates
    selected = max(pool, key=lambda path: (_extract_run_ratio(path), path.name))

    candidate_text = ", ".join(
        "%s(r=%s)" % (
            path.name,
            ("none" if not math.isfinite(_extract_run_ratio(path)) else ("%g" % _extract_run_ratio(path))),
        )
        for path in sorted(candidates, key=lambda p: p.name)
    )
    selected_ratio = _extract_run_ratio(selected)
    print(
        "[run-select] M=%d candidates=[%s] -> selected=%s ratio=%s"
        % (
            int(M),
            candidate_text,
            selected.name,
            ("none" if not math.isfinite(selected_ratio) else ("%g" % selected_ratio)),
        ),
        flush=True,
    )
    return selected


def _cnn_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        runs_root=args.runs_root,
        M=list(args.M),
        methods=("cnn",),
        workflow="forward_then_inverse",
        device=args.device,
        seed=args.seed,
        split_seed=args.split_seed,
        validation_fraction=args.validation_fraction,
        train_label_fraction=1.0,
        label_subset_seed=args.label_subset_seed,
        source_cnn_folder=args.source_cnn_folder,
        output_folder=str(Path(args.output_folder) / "cnn_direct"),
        batch_size=args.cnn_batch_size,
        learning_rate=args.cnn_learning_rate,
        weight_decay=args.cnn_weight_decay,
        max_epochs=args.cnn_max_epochs,
        min_epochs=args.cnn_min_epochs,
        validate_every_epochs=args.cnn_validate_every_epochs,
        patience_epochs=args.cnn_patience_epochs,
        lr_patience_checks=args.cnn_lr_patience_checks,
        lr_reduction_factor=args.lr_reduction_factor,
        min_learning_rate=args.min_learning_rate,
        log_every_epochs=args.log_every_epochs,
        eval_batch_size=args.cnn_eval_batch_size,
        improvement_rel_tol=args.improvement_rel_tol,
        improvement_abs_tol=args.improvement_abs_tol,
        gradient_clip=args.gradient_clip,
        field_loss_weight=args.cnn_field_loss_weight,
        power_loss_weight=args.cnn_power_loss_weight,
        cnn_hidden=args.cnn_hidden,
        cnn_kernel_size=args.cnn_kernel_size,
        cnn_dilations=list(args.cnn_dilations),
        # Unused branch/trunk parameters are supplied because the embedded
        # implementation shares one argument namespace for both old methods.
        fourier_features=8,
        branch_hidden=512,
        branch_layers=4,
        trunk_hidden=128,
        trunk_layers=3,
        latent_dim=128,
        dropout=0.05,
        residual_output=args.cnn_residual_output,
        cache_seen_in_ram=args.cache_seen_in_ram,
        amp=args.amp,
        ssfm_batch_size=args.ssfm_batch_size,
        max_eval_samples=args.max_eval_samples,
        force_train=args.force_train,
        force_eval=args.force_eval,
        hard_max_epochs=args.cnn_hard_max_epochs,
        auto_extend_epochs=args.auto_extend_epochs,
        resume_save_every_epochs=args.resume_save_every_epochs,
        allow_unconverged=args.allow_unconverged,
        final_min_epochs=args.cnn_final_min_epochs,
        final_max_epochs=args.cnn_final_max_epochs,
        final_optimizer_updates=args.cnn_final_optimizer_updates,
        final_learning_rate=args.cnn_final_learning_rate,
        stop_on_error=args.stop_on_error,
    )


def _ddnn_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        runs_root=args.runs_root,
        M=list(args.M),
        stage="all",
        device=args.device,
        seed=args.seed,
        split_seed=args.split_seed,
        validation_fraction=args.validation_fraction,
        train_label_fraction=1.0,
        label_subset_seed=args.label_subset_seed,
        source_cnn_folder=args.source_cnn_folder,
        output_folder=str(Path(args.output_folder) / "ddnn_same_architecture"),
        config_batch_size=args.ddnn_config_batch_size,
        points_per_endpoint=args.ddnn_points_per_endpoint,
        validation_points_per_endpoint=args.ddnn_validation_points_per_endpoint,
        learning_rate=args.ddnn_learning_rate,
        weight_decay=args.ddnn_weight_decay,
        power_loss_weight=args.ddnn_power_loss_weight,
        max_epochs=args.ddnn_max_epochs,
        min_epochs=args.ddnn_min_epochs,
        validate_every_epochs=args.ddnn_validate_every_epochs,
        patience_epochs=args.ddnn_patience_epochs,
        lr_patience_checks=args.ddnn_lr_patience_checks,
        lr_reduction_factor=args.lr_reduction_factor,
        min_learning_rate=args.min_learning_rate,
        improvement_rel_tol=args.improvement_rel_tol,
        improvement_abs_tol=args.improvement_abs_tol,
        gradient_clip=args.gradient_clip,
        validation_chunk_size=args.ddnn_validation_chunk_size,
        log_every_epochs=args.log_every_epochs,
        prediction_chunk_size=args.ddnn_prediction_chunk_size,
        ssfm_batch_size=args.ssfm_batch_size,
        max_eval_samples=args.max_eval_samples,
        inverse_samples=args.inverse_samples,
        inverse_sample_seed=args.inverse_sample_seed,
        restarts=args.restarts,
        inverse_epochs=args.inverse_epochs,
        inverse_learning_rate=args.inverse_learning_rate,
        inverse_min_learning_rate=args.inverse_min_learning_rate,
        terminal_points=args.terminal_points,
        inverse_point_chunk=args.inverse_point_chunk,
        inverse_early_stop_min_epochs=args.inverse_early_stop_min_epochs,
        inverse_early_stop_patience=args.inverse_early_stop_patience,
        inverse_early_stop_fraction=args.inverse_early_stop_fraction,
        inverse_early_stop_rel_delta=args.inverse_early_stop_rel_delta,
        inverse_early_stop_abs_delta=args.inverse_early_stop_abs_delta,
        inverse_log_every=args.inverse_log_every,
        no_amp=not args.amp,
        force_train=args.force_train,
        force_eval=args.force_eval,
        force_inverse=args.force_inverse,
        hard_max_epochs=args.ddnn_hard_max_epochs,
        auto_extend_epochs=args.auto_extend_epochs,
        resume_save_every_epochs=args.resume_save_every_epochs,
        inverse_resume_every_epochs=args.inverse_resume_every_epochs,
        allow_unconverged=args.allow_unconverged,
        final_min_epochs=args.ddnn_final_min_epochs,
        final_max_epochs=args.ddnn_final_max_epochs,
        final_optimizer_updates=args.ddnn_final_optimizer_updates,
        final_learning_rate=args.ddnn_final_learning_rate,
        stop_on_error=args.stop_on_error,
    )


def _round_up_to_multiple(value: int, multiple: int) -> int:
    multiple = max(1, int(multiple))
    value = max(1, int(value))
    return int(math.ceil(value / float(multiple)) * multiple)


def _configure_cnn_epoch_budget(
    cnn: types.ModuleType,
    args: argparse.Namespace,
    run_dir: Path,
    base_args: argparse.Namespace,
) -> argparse.Namespace:
    """Convert an optimizer-update budget into an M-specific epoch budget.

    One epoch still means one complete pass over the selected labeled pairs.
    Small-M sets have only one or a few batches per epoch, so they automatically
    receive more epochs. Large-M sets have many batches per epoch and therefore
    need fewer epochs for a comparable number of optimizer updates.
    """
    configured = copy.copy(base_args)
    if not bool(args.cnn_auto_epoch_by_updates):
        return configured

    seen = cnn.np.asarray(
        cnn.load_combinations_csv(run_dir / "dataset" / "seen_combinations.csv"),
        dtype=cnn.np.float64,
    )
    train_indices, _ = cnn.make_internal_split(
        n_seen=len(seen),
        validation_fraction=float(args.validation_fraction),
        split_seed=int(args.split_seed),
    )
    selected_indices = cnn.select_training_fraction(
        train_indices,
        fraction=float(args.train_label_fraction),
        subset_seed=int(args.label_subset_seed),
    )
    batches_per_epoch = max(
        1,
        int(math.ceil(len(selected_indices) / float(max(1, int(args.cnn_batch_size))))),
    )
    validate_every = max(1, int(args.cnn_validate_every_epochs))

    min_epochs_from_updates = int(
        math.ceil(float(args.cnn_min_optimizer_updates) / batches_per_epoch)
    )
    max_epochs_from_updates = int(
        math.ceil(float(args.cnn_max_optimizer_updates) / batches_per_epoch)
    )
    patience_epochs_from_updates = int(
        math.ceil(float(args.cnn_patience_optimizer_updates) / batches_per_epoch)
    )
    lr_patience_checks = int(
        math.ceil(
            float(args.cnn_lr_patience_optimizer_updates)
            / float(batches_per_epoch * validate_every)
        )
    )

    effective_min_epochs = _round_up_to_multiple(
        max(int(args.cnn_min_epochs), min_epochs_from_updates),
        validate_every,
    )
    effective_patience_epochs = _round_up_to_multiple(
        max(validate_every, patience_epochs_from_updates),
        validate_every,
    )
    effective_max_epochs = _round_up_to_multiple(
        max(effective_min_epochs, max_epochs_from_updates),
        validate_every,
    )
    effective_max_epochs = min(
        int(args.cnn_hard_max_epochs),
        effective_max_epochs,
    )
    if effective_max_epochs < effective_min_epochs:
        raise ValueError(
            "The automatic CNN hard epoch cap is below the required minimum. "
            "Increase --cnn-hard-max-epochs."
        )

    configured.min_epochs = int(effective_min_epochs)
    configured.max_epochs = int(effective_max_epochs)
    configured.hard_max_epochs = int(effective_max_epochs)
    configured.patience_epochs = int(effective_patience_epochs)
    configured.lr_patience_checks = max(1, int(lr_patience_checks))

    print(
        "[auto-epoch] selected_pairs=%d batch_size=%d batches/epoch=%d "
        "min_epochs=%d max_epochs=%d patience_epochs=%d "
        "lr_patience_checks=%d update_budget=[%d,%d]"
        % (
            len(selected_indices),
            int(args.cnn_batch_size),
            batches_per_epoch,
            configured.min_epochs,
            configured.max_epochs,
            configured.patience_epochs,
            configured.lr_patience_checks,
            int(args.cnn_min_optimizer_updates),
            int(args.cnn_max_optimizer_updates),
        ),
        flush=True,
    )
    return configured




def _configure_ddnn_sampling(
    ddnn: types.ModuleType,
    args: argparse.Namespace,
    run_dir: Path,
    base_args: argparse.Namespace,
) -> argparse.Namespace:
    """Resolve DDNN endpoint time-coordinate sampling against the real grid.

    The formal SSFM endpoint dataset contains ``n_time`` unique time samples per
    waveform.  Training uses a fresh random subset for every configuration in
    every epoch.  Validation defaults to all unique time samples, so the
    selected checkpoint is judged on complete endpoint waveforms.

    Asking for more than ``n_time`` points would only sample duplicates with
    replacement and adds computation without new information; V10 rejects it.
    """
    configured = copy.copy(base_args)
    store = ddnn.EndpointLabelStore(run_dir, str(args.source_cnn_folder))
    n_time = int(store.n_time)

    train_points = int(args.ddnn_points_per_endpoint)
    validation_points = int(args.ddnn_validation_points_per_endpoint)
    if train_points == 0:
        train_points = n_time
    if validation_points == 0:
        validation_points = n_time
    if train_points > n_time:
        raise ValueError(
            "DDNN training requested %d points per endpoint, but the stored "
            "waveform has only %d unique time samples. Values above %d only "
            "duplicate coordinates; use at most the full grid."
            % (train_points, n_time, n_time)
        )
    if validation_points > n_time:
        raise ValueError(
            "DDNN validation requested %d points per endpoint, but the stored "
            "waveform has only %d unique time samples."
            % (validation_points, n_time)
        )

    configured.points_per_endpoint = int(train_points)
    configured.validation_points_per_endpoint = int(validation_points)
    coordinates_per_full_batch = (
        int(configured.config_batch_size) * 2 * int(train_points)
    )
    print(
        "[ddnn-sampling] n_time=%d train_points/endpoint/config/epoch=%d "
        "validation_points/endpoint/config=%d resample_train_each_epoch=True "
        "config_batch=%d coordinates/full_update=%d"
        % (
            n_time, train_points, validation_points,
            int(configured.config_batch_size), coordinates_per_full_batch,
        ),
        flush=True,
    )
    return configured


def _configure_ddnn_epoch_budget(
    ddnn: types.ModuleType,
    args: argparse.Namespace,
    run_dir: Path,
    base_args: argparse.Namespace,
) -> argparse.Namespace:
    """Choose a practical, M-specific DDNN selection budget.

    One DDNN epoch still visits every selected seen configuration once.  The
    number of optimizer steps in that epoch is therefore
    ceil(n_selection_configs / config_batch_size).  The fixed update budget
    prevents small-M models from stopping too early and prevents large-M models
    from running thousands of unnecessarily expensive epochs.

    Early stopping remains active.  If it is not triggered before the budget,
    the historical best validation checkpoint is accepted, which is the normal
    fixed-budget model-selection rule for supervised learning.
    """
    configured = copy.copy(base_args)
    if not bool(args.ddnn_auto_epoch_by_updates):
        return configured

    seen = ddnn.np.asarray(
        ddnn.load_combinations_csv(run_dir / "dataset" / "seen_combinations.csv"),
        dtype=ddnn.np.float32,
    )
    selection_ids, _ = ddnn.make_config_split(
        len(seen), float(args.validation_fraction), int(args.split_seed)
    )
    batches_per_epoch = max(
        1,
        int(math.ceil(len(selection_ids) / float(max(1, int(base_args.config_batch_size))))),
    )
    validate_every = max(1, int(args.ddnn_validate_every_epochs))

    min_epochs_from_updates = int(
        math.ceil(float(args.ddnn_min_optimizer_updates) / batches_per_epoch)
    )
    max_epochs_from_updates = int(
        math.ceil(float(args.ddnn_max_optimizer_updates) / batches_per_epoch)
    )
    patience_epochs_from_updates = int(
        math.ceil(float(args.ddnn_patience_optimizer_updates) / batches_per_epoch)
    )
    lr_patience_checks = int(
        math.ceil(
            float(args.ddnn_lr_patience_optimizer_updates)
            / float(batches_per_epoch * validate_every)
        )
    )

    effective_min_epochs = _round_up_to_multiple(
        max(int(args.ddnn_min_epochs), min_epochs_from_updates), validate_every
    )
    effective_max_epochs = _round_up_to_multiple(
        max(effective_min_epochs, max_epochs_from_updates), validate_every
    )
    effective_max_epochs = min(int(args.ddnn_hard_max_epochs), effective_max_epochs)
    if effective_max_epochs < effective_min_epochs:
        raise ValueError(
            "The automatic DDNN hard epoch cap is below the required minimum. "
            "Increase --ddnn-hard-max-epochs."
        )
    effective_patience_epochs = _round_up_to_multiple(
        max(validate_every, patience_epochs_from_updates), validate_every
    )

    configured.min_epochs = int(effective_min_epochs)
    configured.max_epochs = int(effective_max_epochs)
    configured.hard_max_epochs = int(effective_max_epochs)
    configured.patience_epochs = int(effective_patience_epochs)
    configured.lr_patience_checks = max(2, int(lr_patience_checks))

    print(
        "[ddnn-budget] selection_configs=%d config_batch=%d batches/epoch=%d "
        "min_epochs=%d max_epochs=%d patience_epochs=%d lr_patience_checks=%d "
        "update_budget=[%d,%d]"
        % (
            len(selection_ids), int(base_args.config_batch_size), batches_per_epoch,
            configured.min_epochs, configured.max_epochs,
            configured.patience_epochs, configured.lr_patience_checks,
            int(args.ddnn_min_optimizer_updates), int(args.ddnn_max_optimizer_updates),
        ),
        flush=True,
    )
    return configured


def _run_cnn(cnn: types.ModuleType, args: argparse.Namespace, device: Any,
             m_values: Sequence[int], failures: List[Dict[str, Any]]) -> None:
    base_cargs = _cnn_namespace(args)
    do_forward_train = args.workflow in ("all", "forward_only", "train_only")
    do_forward_eval = args.workflow in ("all", "forward_only", "evaluate_only")
    do_inverse_train = args.workflow in ("all", "inverse_only", "train_only")
    do_inverse_eval = args.workflow in ("all", "inverse_only", "evaluate_only")

    for direction, do_train, do_eval in (
        ("forward", do_forward_train, do_forward_eval),
        ("inverse", do_inverse_train, do_inverse_eval),
    ):
        for M in m_values:
            try:
                run_dir = _find_run_dir_highest_ratio(Path(args.runs_root), int(M))
                cargs = _configure_cnn_epoch_budget(cnn, args, run_dir, base_cargs)
                print("\n========== CNN %s | M=%d ==========" % (direction.upper(), M))
                if do_train:
                    cnn.train_one_model(
                        run_dir=run_dir,
                        method="cnn",
                        direction=direction,
                        args=cargs,
                        device=device,
                    )
                if do_eval:
                    cnn.evaluate_one_model(
                        run_dir=run_dir,
                        method="cnn",
                        direction=direction,
                        args=cargs,
                        device=device,
                    )
            except Exception as exc:
                item = {
                    "method": "cnn", "direction": direction, "M": int(M),
                    "stage": "train_eval", "error": repr(exc),
                }
                failures.append(item)
                print("[FAILED] %s" % item)
                if args.stop_on_error:
                    raise


def _run_ddnn(ddnn: types.ModuleType, args: argparse.Namespace, device: Any,
              m_values: Sequence[int], failures: List[Dict[str, Any]]) -> None:
    base_dargs = _ddnn_namespace(args)
    do_train = args.workflow in ("all", "forward_only", "train_only")
    do_eval = args.workflow in ("all", "forward_only", "evaluate_only")
    do_inverse = args.workflow in ("all", "inverse_only", "evaluate_only")

    for M in m_values:
        try:
            run_dir = _find_run_dir_highest_ratio(Path(args.runs_root), int(M))
            sampled_dargs = _configure_ddnn_sampling(ddnn, args, run_dir, base_dargs)
            dargs = _configure_ddnn_epoch_budget(ddnn, args, run_dir, sampled_dargs)
            print("\n========== SAME-ARCH DDNN | M=%d ========== " % M)
            if do_train:
                ddnn.train_model(run_dir, dargs, device)
            if do_eval:
                ddnn.evaluate_forward(run_dir, dargs, device)
            if do_inverse:
                ddnn.run_inverse(run_dir, dargs, device)
        except Exception as exc:
            item = {
                "method": "ddnn_same_architecture", "direction": "forward+inverse",
                "M": int(M), "stage": "workflow", "error": repr(exc),
            }
            failures.append(item)
            print("[FAILED] %s" % item)
            if args.stop_on_error:
                raise


def _collect_unified_summary(root: Path, output_folder: str,
                             m_values: Sequence[int], seed: int) -> Path:
    """Collect summaries from the actual run directory containing output_folder.

    A fixed M may have multiple formal run folders (for example, different
    training fractions).  The previous implementation inspected only the first
    lexicographically sorted M* folder, which could differ from the folder used
    by base.find_run_dir during training.  This implementation scans every M*
    candidate and only reads candidates that actually contain output_folder.
    """
    rows: List[Dict[str, Any]] = []
    seen_keys = set()

    for M in m_values:
        run_dir = _find_run_dir_highest_ratio(root, int(M))
        experiment_root = run_dir / output_folder
        if not experiment_root.exists():
            continue

        cnn_root = experiment_root / "cnn_direct"
        for direction in ("forward", "inverse"):
            path = cnn_root / ("cnn_%s" % direction) / "eval" / "summary.json"
            if not path.exists():
                continue
            key = (int(M), "cnn", direction, str(run_dir.resolve()))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            data = json.loads(path.read_text(encoding="utf-8"))
            row: Dict[str, Any] = {
                "M": int(M),
                "run_dir": run_dir.name,
                "method": "cnn",
                "direction": direction,
                "n_cases": int(data.get("n_cases", 0)),
            }
            for metric, stats in data.get("metrics", {}).items():
                if isinstance(stats, dict) and "mean" in stats:
                    row[metric + "_mean"] = stats["mean"]
                    row[metric + "_std"] = stats.get("std")
            rows.append(row)

        ddnn_root = experiment_root / "ddnn_same_architecture"
        ddnn_paths = (
            ("forward", ddnn_root / "eval" / ("seed_%d" % int(seed)) / "summary.json"),
            ("inverse_amplitude", ddnn_root / "inverse" / ("seed_%d" % int(seed)) / "summary.json"),
        )
        for direction, path in ddnn_paths:
            if not path.exists():
                search_root = ddnn_root / ("eval" if direction == "forward" else "inverse")
                found = sorted(search_root.rglob("summary.json")) if search_root.exists() else []
                path = found[0] if found else path
            if not path.exists():
                continue
            key = (int(M), "ddnn_same_architecture", direction, str(run_dir.resolve()))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            data = json.loads(path.read_text(encoding="utf-8"))
            row = {
                "M": int(M),
                "run_dir": run_dir.name,
                "method": "ddnn_same_architecture",
                "direction": direction,
                "n_cases": int(data.get("n_cases", data.get("n_samples", 0))),
            }
            for metric, stats in data.get("metrics", {}).items():
                if isinstance(stats, dict) and "mean" in stats:
                    row[metric + "_mean"] = stats["mean"]
                    row[metric + "_std"] = stats.get("std")
            rows.append(row)

    output = root / (output_folder + "_unified_summary.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).sort_values(
            ["method", "direction", "M", "run_dir"]
        ).to_csv(output, index=False, encoding="utf-8-sig")
        print("[summary] wrote %d row(s): %s" % (len(rows), output), flush=True)
    else:
        # Write a real CSV even when no per-model summary was found, so the
        # workflow report never points to a silently missing file.
        pd.DataFrame(columns=[
            "M", "run_dir", "method", "direction", "n_cases"
        ]).to_csv(output, index=False, encoding="utf-8-sig")
        print(
            "[warning] no per-model summary.json found under output folder %s; "
            "wrote an empty CSV: %s" % (output_folder, output),
            flush=True,
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "One-file endpoint workflow: direct CNN forward/inverse networks and "
            "same-architecture DDNN forward plus frozen-forward inverse comparison."
        )
    )
    p.add_argument("--runs-root", required=True)
    p.add_argument("--M", nargs="+", type=int, required=True)
    p.add_argument("--methods", nargs="+", choices=("cnn", "ddnn"), default=("cnn", "ddnn"))
    p.add_argument(
        "--workflow",
        choices=("all", "forward_only", "inverse_only", "train_only", "evaluate_only"),
        default="all",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--validation-fraction", type=float, default=0.20)
    p.add_argument(
        "--train-label-fraction",
        type=float,
        default=1.0,
        help=(
            "Formal V7 accepts only 1.0. The 80/20 split is used only for epoch "
            "selection; final retraining always uses every original seen waveform."
        ),
    )
    p.add_argument("--label-subset-seed", type=int, default=4242)
    p.add_argument("--source-cnn-folder", default="cnn_full_propagation_v2")
    p.add_argument("--output-folder", default="endpoint_all_seen_budget_resume_v9")

    # Shared optimization/evaluation controls.
    p.add_argument("--lr-reduction-factor", type=float, default=0.5)
    p.add_argument("--min-learning-rate", type=float, default=1e-6)
    p.add_argument("--improvement-rel-tol", type=float, default=1e-3)
    p.add_argument("--improvement-abs-tol", type=float, default=1e-9)
    p.add_argument("--gradient-clip", type=float, default=5.0)
    p.add_argument("--log-every-epochs", type=int, default=20)
    p.add_argument("--ssfm-batch-size", type=int, default=16)
    p.add_argument("--max-eval-samples", type=int, default=0)
    p.add_argument("--cache-seen-in-ram", action="store_true")
    p.add_argument("--amp", action="store_true")

    # CNN direct forward/inverse networks.
    p.add_argument("--cnn-batch-size", type=int, default=64)
    p.add_argument("--cnn-learning-rate", type=float, default=1e-4)
    p.add_argument("--cnn-weight-decay", type=float, default=1e-6)
    p.add_argument("--cnn-max-epochs", type=int, default=10000, help="Soft epoch boundary; training auto-extends after this when validation is still improving.")
    p.add_argument("--cnn-min-epochs", type=int, default=1000)
    p.add_argument("--cnn-validate-every-epochs", type=int, default=20)
    p.add_argument("--cnn-patience-epochs", type=int, default=500)
    p.add_argument("--cnn-lr-patience-checks", type=int, default=20)
    p.add_argument(
        "--cnn-auto-epoch-by-updates",
        dest="cnn_auto_epoch_by_updates",
        action="store_true",
        help=(
            "Choose an M-specific CNN selection ceiling from optimizer-update "
            "budgets. At the budget ceiling, accept the best validation checkpoint."
        ),
    )
    p.add_argument(
        "--cnn-fixed-epoch-budget",
        dest="cnn_auto_epoch_by_updates",
        action="store_false",
        help="Disable update-budget conversion and use the explicit epoch limits.",
    )
    p.set_defaults(cnn_auto_epoch_by_updates=True)
    p.add_argument("--cnn-min-optimizer-updates", type=int, default=5000)
    p.add_argument("--cnn-max-optimizer-updates", type=int, default=50000)
    p.add_argument("--cnn-patience-optimizer-updates", type=int, default=3000)
    p.add_argument("--cnn-lr-patience-optimizer-updates", type=int, default=1000)
    p.add_argument("--cnn-hard-max-epochs", type=int, default=100000)
    p.add_argument("--cnn-eval-batch-size", type=int, default=128)
    p.add_argument("--cnn-field-loss-weight", type=float, default=1.0)
    p.add_argument("--cnn-power-loss-weight", type=float, default=1.0)
    p.add_argument("--cnn-hidden", type=int, default=64)
    p.add_argument("--cnn-kernel-size", type=int, default=11)
    p.add_argument("--cnn-dilations", nargs="+", type=int, default=(1, 4, 16, 64))
    p.add_argument("--cnn-residual-output", dest="cnn_residual_output", action="store_true")
    p.add_argument("--cnn-no-residual-output", dest="cnn_residual_output", action="store_false")
    p.set_defaults(cnn_residual_output=True)
    p.add_argument("--cnn-final-min-epochs", type=int, default=20, help="Minimum all-seen warm-start fine-tuning epochs.")
    p.add_argument("--cnn-final-max-epochs", type=int, default=1000, help="Maximum all-seen warm-start fine-tuning epochs.")
    p.add_argument("--cnn-final-optimizer-updates", type=int, default=2000, help="Target optimizer updates for all-seen warm-start fine-tuning.")
    p.add_argument("--cnn-final-learning-rate", type=float, default=1e-6, help="Learning rate for all-seen warm-start fine-tuning.")

    # Same-architecture endpoint DDNN forward model.
    p.add_argument("--ddnn-config-batch-size", type=int, default=64)
    p.add_argument("--ddnn-points-per-endpoint", type=int, default=256)
    p.add_argument(
        "--ddnn-validation-points-per-endpoint", type=int, default=0,
        help="Fixed validation points per endpoint and configuration. Use 0 for the complete stored time grid (recommended).",
    )
    p.add_argument("--ddnn-learning-rate", type=float, default=1e-3)
    p.add_argument("--ddnn-weight-decay", type=float, default=0.0)
    p.add_argument("--ddnn-power-loss-weight", type=float, default=0.0)
    p.add_argument("--ddnn-max-epochs", type=int, default=10000, help="Soft epoch boundary; training auto-extends after this when validation is still improving.")
    p.add_argument("--ddnn-hard-max-epochs", type=int, default=30000, help="Absolute DDNN selection cap. No final model is accepted before convergence unless --allow-unconverged is set.")
    p.add_argument("--ddnn-min-epochs", type=int, default=500)
    p.add_argument("--ddnn-validate-every-epochs", type=int, default=10)
    p.add_argument("--ddnn-patience-epochs", type=int, default=300)
    p.add_argument("--ddnn-lr-patience-checks", type=int, default=10)
    p.add_argument(
        "--ddnn-auto-epoch-by-updates",
        dest="ddnn_auto_epoch_by_updates",
        action="store_true",
        help=(
            "Choose an M-specific DDNN selection ceiling from optimizer-update "
            "budgets. At the budget ceiling, accept the best validation checkpoint."
        ),
    )
    p.add_argument(
        "--ddnn-fixed-epoch-budget",
        dest="ddnn_auto_epoch_by_updates",
        action="store_false",
        help="Disable DDNN update-budget conversion and use explicit epoch limits.",
    )
    p.set_defaults(ddnn_auto_epoch_by_updates=True)
    p.add_argument("--ddnn-min-optimizer-updates", type=int, default=4000)
    p.add_argument("--ddnn-max-optimizer-updates", type=int, default=20000)
    p.add_argument("--ddnn-patience-optimizer-updates", type=int, default=3000)
    p.add_argument("--ddnn-lr-patience-optimizer-updates", type=int, default=1000)
    p.add_argument("--ddnn-validation-chunk-size", type=int, default=131072)
    p.add_argument("--ddnn-prediction-chunk-size", type=int, default=131072)
    p.add_argument("--ddnn-final-min-epochs", type=int, default=20, help="Minimum all-seen warm-start fine-tuning epochs.")
    p.add_argument("--ddnn-final-max-epochs", type=int, default=1000, help="Maximum all-seen warm-start fine-tuning epochs.")
    p.add_argument("--ddnn-final-optimizer-updates", type=int, default=3000, help="Target optimizer updates for all-seen warm-start fine-tuning.")
    p.add_argument("--ddnn-final-learning-rate", type=float, default=1e-5, help="Learning rate for all-seen warm-start fine-tuning.")

    # DDNN frozen-forward inverse amplitude reconstruction.
    p.add_argument("--inverse-samples", type=int, default=10, help="Number of unseen inverse cases; use 0 for every unseen case (usually very expensive).")
    p.add_argument("--inverse-sample-seed", type=int, default=2026)
    p.add_argument("--restarts", type=int, default=40)
    p.add_argument("--inverse-epochs", type=int, default=3000)
    p.add_argument("--inverse-learning-rate", type=float, default=3e-2)
    p.add_argument("--inverse-min-learning-rate", type=float, default=5e-4)
    p.add_argument("--terminal-points", type=int, default=512)
    p.add_argument("--inverse-point-chunk", type=int, default=65536)
    p.add_argument("--inverse-early-stop-min-epochs", type=int, default=500)
    p.add_argument("--inverse-early-stop-patience", type=int, default=300)
    p.add_argument("--inverse-early-stop-fraction", type=float, default=0.90)
    p.add_argument("--inverse-early-stop-rel-delta", type=float, default=1e-4)
    p.add_argument("--inverse-early-stop-abs-delta", type=float, default=1e-8)
    p.add_argument("--inverse-log-every", type=int, default=100)

    p.add_argument("--auto-extend-epochs", type=int, default=5000, help="Print an automatic extension boundary every N epochs after the soft maximum.")
    p.add_argument("--resume-save-every-epochs", type=int, default=10, help="Save final-training resume state every N epochs. Selection state is saved at every validation.")
    p.add_argument("--inverse-resume-every-epochs", type=int, default=100, help="Save DDNN inverse-optimization resume state every N epochs.")
    p.add_argument("--allow-unconverged", action="store_true", help="Legacy compatibility flag. V9 already accepts the historical best validation checkpoint at the fixed selection budget.")

    p.add_argument("--force-train", action="store_true")
    p.add_argument("--force-eval", action="store_true")
    p.add_argument("--force-inverse", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    return p


def _validate_args(args: argparse.Namespace) -> None:
    positive_names = (
        "cnn_batch_size", "cnn_max_epochs", "cnn_min_epochs",
        "cnn_min_optimizer_updates", "cnn_max_optimizer_updates",
        "cnn_patience_optimizer_updates", "cnn_lr_patience_optimizer_updates",
        "cnn_hard_max_epochs", "cnn_final_min_epochs", "cnn_final_max_epochs",
        "cnn_final_optimizer_updates", "cnn_final_learning_rate",
        "ddnn_config_batch_size", "ddnn_points_per_endpoint",
        "ddnn_max_epochs",
        "ddnn_hard_max_epochs", "ddnn_min_epochs",
        "ddnn_min_optimizer_updates", "ddnn_max_optimizer_updates",
        "ddnn_patience_optimizer_updates", "ddnn_lr_patience_optimizer_updates",
        "ddnn_final_min_epochs",
        "ddnn_final_max_epochs", "ddnn_final_optimizer_updates", "ddnn_final_learning_rate",
        "ssfm_batch_size", "gradient_clip",
        "auto_extend_epochs", "resume_save_every_epochs", "inverse_resume_every_epochs",
    )
    for name in positive_names:
        if float(getattr(args, name)) <= 0:
            raise ValueError("--%s must be positive." % name.replace("_", "-"))
    if int(args.ddnn_validation_points_per_endpoint) < 0:
        raise ValueError("--ddnn-validation-points-per-endpoint must be 0 or positive.")
    if (not args.cnn_auto_epoch_by_updates) and args.cnn_min_epochs > args.cnn_max_epochs:
        raise ValueError("CNN min epochs cannot exceed max epochs.")
    if args.cnn_min_optimizer_updates > args.cnn_max_optimizer_updates:
        raise ValueError("CNN minimum optimizer updates cannot exceed maximum updates.")
    if args.ddnn_min_optimizer_updates > args.ddnn_max_optimizer_updates:
        raise ValueError("DDNN minimum optimizer updates cannot exceed maximum updates.")
    if args.ddnn_min_epochs > args.ddnn_max_epochs and not args.ddnn_auto_epoch_by_updates:
        raise ValueError("DDNN min epochs cannot exceed soft max epochs.")
    if args.cnn_final_min_epochs > args.cnn_final_max_epochs:
        raise ValueError("CNN final min epochs cannot exceed final max epochs.")
    if args.ddnn_final_min_epochs > args.ddnn_final_max_epochs:
        raise ValueError("DDNN final min epochs cannot exceed final max epochs.")
    if args.cnn_max_epochs > args.cnn_hard_max_epochs:
        raise ValueError("CNN soft max epochs cannot exceed hard max epochs.")
    if args.ddnn_max_epochs > args.ddnn_hard_max_epochs:
        raise ValueError("DDNN soft max epochs cannot exceed hard max epochs.")
    if not (0.0 < args.lr_reduction_factor < 1.0):
        raise ValueError("--lr-reduction-factor must be in (0,1).")
    if abs(float(args.train_label_fraction) - 1.0) > 1e-12:
        raise ValueError("V7 formal workflow only supports --train-label-fraction 1.0. Selection uses 80/20 internally, then final retraining uses every seen waveform.")


# =============================================================================
# V7 all-seen retraining, adaptive convergence, and true resume patches
# =============================================================================

def _v7_checkpoint_fallbacks(path: Path) -> List[Path]:
    """Return valid versioned fallbacks for a checkpoint, newest first."""
    patterns = (
        f"{path.name}.fallback_*.pt",
        f"{path.name}.unswapped_pid*.pt",
    )
    items: List[Path] = []
    for pattern in patterns:
        try:
            items.extend(candidate for candidate in path.parent.glob(pattern) if candidate.is_file())
        except OSError:
            pass
    def _mtime(candidate: Path) -> int:
        try:
            return candidate.stat().st_mtime_ns
        except OSError:
            return -1
    return sorted(set(items), key=_mtime, reverse=True)


def _v7_cleanup_checkpoint_fallbacks(path: Path, keep: int = 0) -> None:
    """Best-effort cleanup; never fail training because Windows refuses deletion."""
    for candidate in _v7_checkpoint_fallbacks(path)[max(0, int(keep)):]:
        try:
            candidate.unlink(missing_ok=True)
        except OSError:
            pass


def _v7_atomic_torch_save(torch_module: Any, payload: Dict[str, Any], path: Path) -> None:
    """Save without letting a transient Windows lock terminate training.

    Normal case: write a process-unique temporary file and atomically replace the
    canonical checkpoint.  If Windows keeps the canonical file locked, preserve
    the fully written payload as a versioned fallback and continue training.
    ``_v7_torch_load`` automatically chooses the newest canonical/fallback file.
    """
    import os
    import stat
    import time
    import uuid

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.pid{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    torch_module.save(payload, str(temporary))

    last_error: Optional[BaseException] = None
    for attempt in range(20):
        try:
            if path.exists():
                try:
                    os.chmod(str(path), stat.S_IREAD | stat.S_IWRITE)
                except OSError:
                    pass
            os.replace(str(temporary), str(path))
            _v7_cleanup_checkpoint_fallbacks(path, keep=0)
            return
        except OSError as exc:
            winerror = getattr(exc, "winerror", None)
            if isinstance(exc, PermissionError) or winerror in {5, 32, 33}:
                last_error = exc
                time.sleep(min(0.20 * (attempt + 1), 2.0))
                continue
            try:
                temporary.unlink(missing_ok=True)
            finally:
                raise

    # The checkpoint payload is already completely written. Move it to a unique
    # fallback instead of crashing the multi-hour workflow. On the next save or
    # restart, the loader automatically selects the newest checkpoint by mtime.
    fallback = path.with_name(
        f"{path.name}.fallback_{time.time_ns()}_pid{os.getpid()}.pt"
    )
    try:
        os.replace(str(temporary), str(fallback))
    except OSError:
        # The unique temporary itself is a valid torch checkpoint. Keep it and
        # include it in the warning, although this branch should be very rare.
        fallback = temporary

    print(
        f"[checkpoint-warning] Windows kept the canonical checkpoint locked after "
        f"20 retries; training will continue. Latest state preserved at: {fallback}. "
        f"Original error: {last_error!r}",
        flush=True,
    )
    _v7_cleanup_checkpoint_fallbacks(path, keep=3)


def _v7_torch_load(torch_module: Any, path: Path, map_location: Any) -> Dict[str, Any]:
    candidates: List[Path] = []
    if path.is_file():
        candidates.append(path)
    candidates.extend(_v7_checkpoint_fallbacks(path))

    if not candidates:
        raise FileNotFoundError(f"No checkpoint found for {path}")

    def _mtime(candidate: Path) -> int:
        try:
            return candidate.stat().st_mtime_ns
        except OSError:
            return -1

    selected = max(candidates, key=_mtime)
    if selected != path:
        print(
            f"[resume-fallback] canonical checkpoint was older/locked; loading newest "
            f"saved state: {selected.name}",
            flush=True,
        )
    try:
        return torch_module.load(str(selected), map_location=map_location, weights_only=False)
    except TypeError:
        return torch_module.load(str(selected), map_location=map_location)

def _v7_capture_torch_rng(torch_module: Any) -> Dict[str, Any]:
    # Keep RNG tensors on CPU so that checkpoints remain portable across
    # CPU/CUDA map_location choices and PyTorch versions.
    state: Dict[str, Any] = {
        "cpu": torch_module.get_rng_state().detach().cpu().to(dtype=torch_module.uint8).contiguous().clone()
    }
    if torch_module.cuda.is_available():
        state["cuda"] = [
            item.detach().cpu().to(dtype=torch_module.uint8).contiguous().clone()
            for item in torch_module.cuda.get_rng_state_all()
        ]
    return state


def _v7_rng_as_cpu_byte_tensor(torch_module: Any, value: Any) -> Any:
    """Normalize a serialized RNG state to the CPU ByteTensor required by PyTorch."""
    if value is None:
        return None
    if torch_module.is_tensor(value):
        return value.detach().to(device="cpu", dtype=torch_module.uint8).contiguous()
    if isinstance(value, (bytes, bytearray)):
        return torch_module.tensor(list(value), dtype=torch_module.uint8, device="cpu")
    return torch_module.as_tensor(value, dtype=torch_module.uint8, device="cpu").contiguous()


def _v7_restore_torch_rng(torch_module: Any, state: Dict[str, Any]) -> None:
    if not state:
        return

    # torch.load(..., map_location=cuda) can move the saved CPU RNG tensor to
    # CUDA. torch.set_rng_state strictly requires a CPU torch.ByteTensor, so
    # always normalize it before restoring. This also keeps old V7 checkpoints
    # fully resumable.
    cpu_state = _v7_rng_as_cpu_byte_tensor(torch_module, state.get("cpu"))
    if cpu_state is not None:
        torch_module.set_rng_state(cpu_state)

    if torch_module.cuda.is_available() and state.get("cuda") is not None:
        cuda_states = state.get("cuda")
        if not isinstance(cuda_states, (list, tuple)):
            cuda_states = [cuda_states]
        device_count = int(torch_module.cuda.device_count())
        for device_index, saved_state in enumerate(cuda_states[:device_count]):
            normalized = _v7_rng_as_cpu_byte_tensor(torch_module, saved_state)
            if normalized is not None:
                torch_module.cuda.set_rng_state(normalized, device=device_index)


def _v7_file_sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _v7_check_seen_unseen_split(module: Any, run_dir: Path, M: int) -> Dict[str, Any]:
    np_module = module.np
    seen_path = run_dir / "dataset" / "seen_combinations.csv"
    unseen_path = run_dir / "dataset" / "unseen_combinations.csv"
    seen = np_module.asarray(module.load_combinations_csv(seen_path), dtype=np_module.float64)
    unseen = np_module.asarray(module.load_combinations_csv(unseen_path), dtype=np_module.float64)
    if seen.ndim != 2 or unseen.ndim != 2:
        raise RuntimeError("Seen/unseen combinations must be two-dimensional arrays.")
    if seen.shape[1] != int(M) or unseen.shape[1] != int(M):
        raise RuntimeError("Seen/unseen M does not match the selected run directory.")
    seen_keys = {tuple(float(x) for x in row) for row in seen.tolist()}
    unseen_keys = {tuple(float(x) for x in row) for row in unseen.tolist()}
    if len(seen_keys) != len(seen) or len(unseen_keys) != len(unseen):
        raise RuntimeError("Duplicate combinations were found in the seen or unseen CSV.")
    overlap = seen_keys.intersection(unseen_keys)
    if overlap:
        raise RuntimeError("Seen and unseen combinations overlap: %s" % (next(iter(overlap)),))
    expected_total = int(4 ** int(M))
    if len(seen) + len(unseen) != expected_total:
        raise RuntimeError(
            "Seen + unseen count is %d, but 4^M is %d for M=%d."
            % (len(seen) + len(unseen), expected_total, int(M))
        )
    return {
        "M": int(M),
        "n_seen": int(len(seen)),
        "n_unseen": int(len(unseen)),
        "total": int(len(seen) + len(unseen)),
        "expected_total": int(expected_total),
        "seen_csv": str(seen_path),
        "unseen_csv": str(unseen_path),
        "seen_sha256": _v7_file_sha256(seen_path),
        "unseen_sha256": _v7_file_sha256(unseen_path),
        "no_overlap": True,
        "same_files_as_pinn": True,
    }


def _v7_save_selection_progress(
    module: Any,
    path: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    best_epoch: int,
    best_loss: float,
    last_loss: float,
    history: List[Dict[str, Any]],
    lr_schedule: List[float],
    numpy_rng: Any,
    elapsed_sec: float,
    model_config: Dict[str, Any],
) -> None:
    _v7_atomic_torch_save(
        module.torch,
        {
            "kind": "selection_resume_v7",
            "epoch": int(epoch),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_epoch": int(best_epoch),
            "best_loss": float(best_loss),
            "last_loss": float(last_loss),
            "history": history,
            "lr_schedule": lr_schedule,
            "numpy_rng_state": numpy_rng.bit_generator.state,
            "torch_rng_state": _v7_capture_torch_rng(module.torch),
            "elapsed_sec": float(elapsed_sec),
            "model_config": model_config,
        },
        path,
    )


def _v7_save_final_progress(
    module: Any,
    path: Path,
    model: Any,
    optimizer: Any,
    scaler: Any,
    epoch: int,
    history: List[Dict[str, Any]],
    numpy_rng: Any,
    elapsed_sec: float,
    model_config: Dict[str, Any],
) -> None:
    _v7_atomic_torch_save(
        module.torch,
        {
            "kind": "final_resume_v7",
            "epoch": int(epoch),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "history": history,
            "numpy_rng_state": numpy_rng.bit_generator.state,
            "torch_rng_state": _v7_capture_torch_rng(module.torch),
            "elapsed_sec": float(elapsed_sec),
            "model_config": model_config,
        },
        path,
    )


def _v7_cnn_train_one_model(
    cnn: Any,
    *,
    run_dir: Path,
    method: str,
    direction: str,
    args: argparse.Namespace,
    device: Any,
) -> Path:
    np_module = cnn.np
    torch_module = cnn.torch
    seen = np_module.asarray(
        cnn.load_combinations_csv(run_dir / "dataset" / "seen_combinations.csv"),
        dtype=np_module.float64,
    )
    unseen = np_module.asarray(
        cnn.load_combinations_csv(run_dir / "dataset" / "unseen_combinations.csv"),
        dtype=np_module.float64,
    )
    split_info = _v7_check_seen_unseen_split(cnn, run_dir, int(seen.shape[1]))
    store = cnn.EndpointPairStore(
        run_dir=run_dir,
        source_cnn_folder=str(args.source_cnn_folder),
        cache_in_ram=bool(args.cache_seen_in_ram),
    )
    if len(seen) != store.n_seen:
        raise RuntimeError("Seen CSV count does not match endpoint dataset count.")

    selection_train_indices, validation_indices = cnn.make_internal_split(
        n_seen=store.n_seen,
        validation_fraction=float(args.validation_fraction),
        split_seed=int(args.split_seed),
    )
    all_seen_indices = np_module.arange(store.n_seen, dtype=np_module.int64)

    experiment_root = run_dir / str(args.output_folder)
    model_root = experiment_root / ("%s_%s" % (method, direction))
    seed_root = model_root / "train" / ("seed_%d" % int(args.seed))
    selection_dir = seed_root / "model_selection"
    final_dir = seed_root / "final"
    final_checkpoint = final_dir / "final_model.pt"
    selection_resume = selection_dir / "selection_resume.pt"
    final_resume = final_dir / "final_resume.pt"
    selection_summary_path = selection_dir / "selection_summary.json"
    best_checkpoint = selection_dir / "best_selection_model.pt"

    if final_checkpoint.exists() and not bool(args.force_train):
        print("[train] exists, skip: %s" % final_checkpoint)
        del store
        return final_checkpoint
    if bool(args.force_train) and seed_root.exists():
        import shutil
        shutil.rmtree(seed_root)
    selection_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    cnn.write_json(model_root / "experiment_manifest.json", {
        "script_version": SCRIPT_VERSION,
        "M": int(seen.shape[1]),
        "run_dir": str(run_dir),
        "method": method,
        "direction": direction,
        "same_seen_unseen_csv_as_pinn": True,
        "split_integrity": split_info,
        "n_seen_pairs": int(len(seen)),
        "n_unseen_pairs": int(len(unseen)),
        "model_selection_only": {
            "training_pairs": int(len(selection_train_indices)),
            "validation_pairs": int(len(validation_indices)),
            "validation_fraction": float(args.validation_fraction),
            "split_seed": int(args.split_seed),
        },
        "final_training_pairs": int(len(all_seen_indices)),
        "final_uses_every_seen_pair": True,
        "unseen_used_for_model_selection": False,
        "training_planes_read": [0, int(store.n_planes_in_source_file - 1)],
        "intermediate_planes_used": 0,
        "resume_supported": True,
        "convergence_required_before_final": False,
        "selection_accepts_best_checkpoint_at_budget": True,
    })
    cnn.write_json(model_root / "split_indices.json", {
        "selection_training_indices": selection_train_indices.tolist(),
        "selection_validation_indices": validation_indices.tolist(),
        "final_training_indices": all_seen_indices.tolist(),
        "final_training_is_all_seen": True,
        "split_seed": int(args.split_seed),
        "unseen_indices_never_used_for_training_or_selection": True,
    })

    amp_enabled = bool(args.amp and device.type == "cuda")
    model_config: Dict[str, Any]
    best_epoch = 0
    best_validation_loss = float("inf")
    last_validation_loss = float("nan")
    stopped_early = False
    history: List[Dict[str, Any]] = []
    lr_schedule: List[float] = []

    if selection_summary_path.exists() and not bool(args.force_train):
        selection_summary = cnn.read_json(selection_summary_path)
        if str(selection_summary.get("convergence_status")) == "EARLY_STOP_CONVERGED" or bool(args.allow_unconverged):
            best_epoch = int(selection_summary["best_epoch"])
            best_validation_loss = float(selection_summary["best_validation_loss"])
            stopped_early = bool(selection_summary.get("stopped_early", False))
            lr_frame = pd.read_csv(selection_dir / "learning_rate_schedule.csv")
            lr_column = "learning_rate_used" if "learning_rate_used" in lr_frame.columns else "learning_rate"
            lr_schedule = [float(x) for x in lr_frame[lr_column].tolist()]
            print("[selection] completed previously; best_epoch=%d" % best_epoch)
        else:
            selection_summary_path.unlink()

    if best_epoch <= 0:
        cnn.set_seed(int(args.seed))
        model, model_config = cnn.build_model(method, store.tau, args)
        model = model.to(device)
        optimizer = torch_module.optim.Adam(
            model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
        )
        scheduler = torch_module.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(args.lr_reduction_factor),
            patience=int(args.lr_patience_checks),
            min_lr=float(args.min_learning_rate),
        )
        scaler = torch_module.cuda.amp.GradScaler(enabled=amp_enabled)
        rng = np_module.random.default_rng(
            int(args.seed) + (100000 if method == "ddnn" else 0) + (10000 if direction == "inverse" else 0)
        )
        start_epoch = 1
        elapsed_before = 0.0

        if selection_resume.exists() and not bool(args.force_train):
            payload = _v7_torch_load(torch_module, selection_resume, device)
            model.load_state_dict(payload["model_state"])
            optimizer.load_state_dict(payload["optimizer_state"])
            scheduler.load_state_dict(payload["scheduler_state"])
            scaler.load_state_dict(payload.get("scaler_state", {}))
            best_epoch = int(payload.get("best_epoch", 0))
            best_validation_loss = float(payload.get("best_loss", float("inf")))
            last_validation_loss = float(payload.get("last_loss", float("nan")))
            history = list(payload.get("history", []))
            lr_schedule = [float(x) for x in payload.get("lr_schedule", [])]
            rng.bit_generator.state = payload["numpy_rng_state"]
            _v7_restore_torch_rng(torch_module, payload.get("torch_rng_state", {}))
            start_epoch = int(payload["epoch"]) + 1
            elapsed_before = float(payload.get("elapsed_sec", 0.0))
            model_config = dict(payload.get("model_config", model_config))
            print("[resume-selection] %s-%s from epoch %d" % (method, direction, start_epoch))

        hard_max = int(args.hard_max_epochs)
        soft_max = int(args.max_epochs)
        extension = max(1, int(args.auto_extend_epochs))
        start_time = cnn.time.perf_counter()
        stop_epoch = start_epoch - 1
        print(
            "[selection] method=%s direction=%s params=%d train=%d val=%d "
            "soft_max=%d hard_max=%d patience=%d resume=%s"
            % (
                method, direction, cnn.count_parameters(model), len(selection_train_indices),
                len(validation_indices), soft_max, hard_max, int(args.patience_epochs),
                str(selection_resume.exists()),
            )
        )

        for epoch in range(start_epoch, hard_max + 1):
            if epoch == soft_max + 1 or (
                epoch > soft_max + 1 and (epoch - soft_max - 1) % extension == 0
            ):
                print(
                    "[auto-extend] %s-%s continues automatically: epoch %d, hard cap %d"
                    % (method, direction, epoch, hard_max),
                    flush=True,
                )
            lr_used = float(optimizer.param_groups[0]["lr"])
            lr_schedule.append(lr_used)
            model.train()
            totals = {"loss": 0.0, "field_rel_sq": 0.0, "power_rel_sq": 0.0}
            count = 0
            n_batches = 0
            for selected in cnn.iter_epoch_batches(
                selection_train_indices, int(args.batch_size), rng, shuffle=True
            ):
                source_np, target_np = store.batch(selected, direction)
                source = torch_module.from_numpy(source_np).to(device)
                target = torch_module.from_numpy(target_np).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch_module.cuda.amp.autocast(enabled=amp_enabled):
                    prediction = model(source)
                    parts = cnn.supervised_endpoint_loss(
                        prediction, target, float(args.field_loss_weight), float(args.power_loss_weight)
                    )
                if not bool(torch_module.isfinite(parts["loss"]).all().item()):
                    raise FloatingPointError("Non-finite CNN loss at epoch %d." % epoch)
                scaler.scale(parts["loss"]).backward()
                scaler.unscale_(optimizer)
                torch_module.nn.utils.clip_grad_norm_(model.parameters(), float(args.gradient_clip))
                scaler.step(optimizer)
                scaler.update()
                batch_count = int(len(selected))
                for key in totals:
                    totals[key] += float(parts[key].detach().cpu()) * batch_count
                count += batch_count
                n_batches += 1
                del source, target, prediction, parts
            average = {key: value / max(count, 1) for key, value in totals.items()}

            should_validate = (
                epoch == 1
                or epoch % int(args.validate_every_epochs) == 0
                or epoch == hard_max
            )
            if not should_validate:
                continue
            validation = cnn.validation_metrics(
                model=model,
                store=store,
                indices=validation_indices,
                direction=direction,
                device=device,
                batch_size=max(1, min(int(args.eval_batch_size), len(validation_indices))),
                field_weight=float(args.field_loss_weight),
                power_weight=float(args.power_loss_weight),
                amp_enabled=amp_enabled,
            )
            last_validation_loss = float(validation["loss"])
            if not np_module.isfinite(last_validation_loss):
                raise FloatingPointError("Non-finite CNN validation loss.")
            scheduler.step(last_validation_loss)
            required = max(
                float(args.improvement_abs_tol),
                float(args.improvement_rel_tol) * abs(best_validation_loss)
                if np_module.isfinite(best_validation_loss) else 0.0,
            )
            improved = (
                not np_module.isfinite(best_validation_loss)
                or last_validation_loss < best_validation_loss - required
            )
            if improved:
                best_validation_loss = last_validation_loss
                best_epoch = int(epoch)
                cnn.save_checkpoint(
                    best_checkpoint,
                    model,
                    model_config,
                    {
                        "phase": "selection",
                        "direction": direction,
                        "best_epoch": best_epoch,
                        "best_validation_loss": best_validation_loss,
                    },
                )
            stale_epochs = int(epoch - best_epoch) if best_epoch > 0 else int(epoch)
            elapsed = elapsed_before + float(cnn.time.perf_counter() - start_time)
            history.append({
                "epoch": int(epoch),
                "batches": int(n_batches),
                "train_pairs_seen": int(count),
                "train_loss": float(average["loss"]),
                "train_field_rel_sq": float(average["field_rel_sq"]),
                "train_power_rel_sq": float(average["power_rel_sq"]),
                "validation_loss": float(validation["loss"]),
                "validation_field_rel_sq": float(validation["field_rel_sq"]),
                "validation_power_rel_sq": float(validation["power_rel_sq"]),
                "best_epoch": int(best_epoch),
                "stale_epochs": int(stale_epochs),
                "learning_rate_used": float(lr_used),
                "learning_rate_next": float(optimizer.param_groups[0]["lr"]),
                "elapsed_sec": float(elapsed),
            })
            pd.DataFrame(history).to_csv(
                selection_dir / "selection_history.csv", index=False, encoding="utf-8-sig"
            )
            pd.DataFrame({
                "epoch": np_module.arange(1, len(lr_schedule) + 1, dtype=np_module.int64),
                "learning_rate_used": np_module.asarray(lr_schedule, dtype=np_module.float64),
            }).to_csv(
                selection_dir / "learning_rate_schedule.csv", index=False, encoding="utf-8-sig"
            )
            _v7_save_selection_progress(
                cnn, selection_resume, model, optimizer, scheduler, scaler,
                epoch, best_epoch, best_validation_loss, last_validation_loss,
                history, lr_schedule, rng, elapsed, model_config,
            )
            print(
                "[select] %s-%s epoch=%d train=%.4e val=%.4e best=%d stale=%d/%d lr=%.2e"
                % (
                    method, direction, epoch, float(average["loss"]), last_validation_loss,
                    best_epoch, stale_epochs, int(args.patience_epochs), lr_used,
                ),
                flush=True,
            )
            stop_epoch = int(epoch)
            if (
                epoch >= int(args.min_epochs)
                and best_epoch > 0
                and stale_epochs >= int(args.patience_epochs)
            ):
                stopped_early = True
                print("[select] early-stop convergence at epoch %d; best=%d" % (epoch, best_epoch))
                break

        convergence_status = "EARLY_STOP_CONVERGED" if stopped_early else "HARD_CAP_REACHED_NOT_ACCEPTED"
        cnn.write_json(selection_summary_path, {
            "script_version": SCRIPT_VERSION,
            "direction": direction,
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best_validation_loss),
            "last_validation_loss": float(last_validation_loss),
            "stop_epoch": int(stop_epoch),
            "stopped_early": bool(stopped_early),
            "convergence_status": convergence_status,
            "n_seen_pairs": int(store.n_seen),
            "n_selection_training_pairs": int(len(selection_train_indices)),
            "n_selection_validation_pairs": int(len(validation_indices)),
            "n_final_training_pairs": int(len(all_seen_indices)),
            "final_uses_all_seen_pairs": True,
            "min_epochs": int(args.min_epochs),
            "soft_max_epochs": int(args.max_epochs),
            "hard_max_epochs": int(args.hard_max_epochs),
            "patience_epochs": int(args.patience_epochs),
            "validate_every_epochs": int(args.validate_every_epochs),
            "parameter_count": int(cnn.count_parameters(model)),
            "model_config": model_config,
        })
        if not stopped_early and not bool(args.allow_unconverged):
            raise RuntimeError(
                "CNN %s M=%d reached hard cap %d without the required early-stop convergence. "
                "The resume checkpoint was kept and no final model was accepted."
                % (direction, int(seen.shape[1]), int(args.hard_max_epochs))
            )
    else:
        model_config = dict(cnn.read_json(selection_summary_path)["model_config"])

    if best_epoch <= 0:
        raise RuntimeError("No valid best epoch is available for final retraining.")
    if len(lr_schedule) < best_epoch:
        lr_frame = pd.read_csv(selection_dir / "learning_rate_schedule.csv")
        lr_column = "learning_rate_used" if "learning_rate_used" in lr_frame.columns else "learning_rate"
        lr_schedule = [float(x) for x in lr_frame[lr_column].tolist()]
    if len(lr_schedule) < best_epoch:
        raise RuntimeError("Saved learning-rate schedule is shorter than best_epoch.")

    cnn.set_seed(int(args.seed))
    final_model, model_config = cnn.build_model(method, store.tau, args)
    final_model = final_model.to(device)
    final_optimizer = torch_module.optim.Adam(
        final_model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    final_scaler = torch_module.cuda.amp.GradScaler(enabled=amp_enabled)
    final_rng = np_module.random.default_rng(
        int(args.seed) + 500000 + (100000 if method == "ddnn" else 0) + (10000 if direction == "inverse" else 0)
    )
    final_history: List[Dict[str, Any]] = []
    final_start_epoch = 1
    final_elapsed_before = 0.0
    if final_resume.exists() and not bool(args.force_train):
        payload = _v7_torch_load(torch_module, final_resume, device)
        final_model.load_state_dict(payload["model_state"])
        final_optimizer.load_state_dict(payload["optimizer_state"])
        final_scaler.load_state_dict(payload.get("scaler_state", {}))
        final_history = list(payload.get("history", []))
        final_rng.bit_generator.state = payload["numpy_rng_state"]
        _v7_restore_torch_rng(torch_module, payload.get("torch_rng_state", {}))
        final_start_epoch = int(payload["epoch"]) + 1
        final_elapsed_before = float(payload.get("elapsed_sec", 0.0))
        print("[resume-final] %s-%s from epoch %d/%d" % (method, direction, final_start_epoch, best_epoch))

    print(
        "[final-all-seen] method=%s direction=%s epochs=%d all_seen_pairs=%d batches/epoch=%d"
        % (
            method, direction, best_epoch, len(all_seen_indices),
            int(math.ceil(len(all_seen_indices) / float(max(1, int(args.batch_size))))),
        )
    )
    final_start = cnn.time.perf_counter()
    for epoch in range(final_start_epoch, best_epoch + 1):
        replay_lr = float(lr_schedule[epoch - 1])
        for group in final_optimizer.param_groups:
            group["lr"] = replay_lr
        final_model.train()
        totals = {"loss": 0.0, "field_rel_sq": 0.0, "power_rel_sq": 0.0}
        count = 0
        n_batches = 0
        for selected in cnn.iter_epoch_batches(all_seen_indices, int(args.batch_size), final_rng, shuffle=True):
            source_np, target_np = store.batch(selected, direction)
            source = torch_module.from_numpy(source_np).to(device)
            target = torch_module.from_numpy(target_np).to(device)
            final_optimizer.zero_grad(set_to_none=True)
            with torch_module.cuda.amp.autocast(enabled=amp_enabled):
                prediction = final_model(source)
                parts = cnn.supervised_endpoint_loss(
                    prediction, target, float(args.field_loss_weight), float(args.power_loss_weight)
                )
            if not bool(torch_module.isfinite(parts["loss"]).all().item()):
                raise FloatingPointError("Non-finite final CNN loss at epoch %d." % epoch)
            final_scaler.scale(parts["loss"]).backward()
            final_scaler.unscale_(final_optimizer)
            torch_module.nn.utils.clip_grad_norm_(final_model.parameters(), float(args.gradient_clip))
            final_scaler.step(final_optimizer)
            final_scaler.update()
            batch_count = int(len(selected))
            for key in totals:
                totals[key] += float(parts[key].detach().cpu()) * batch_count
            count += batch_count
            n_batches += 1
            del source, target, prediction, parts
        averages = {key: value / max(count, 1) for key, value in totals.items()}
        elapsed = final_elapsed_before + float(cnn.time.perf_counter() - final_start)
        if epoch == 1 or epoch % int(args.log_every_epochs) == 0 or epoch == best_epoch:
            final_history.append({
                "epoch": int(epoch),
                "batches": int(n_batches),
                "all_seen_pairs_seen": int(count),
                "train_loss": float(averages["loss"]),
                "train_field_rel_sq": float(averages["field_rel_sq"]),
                "train_power_rel_sq": float(averages["power_rel_sq"]),
                "learning_rate": float(replay_lr),
                "elapsed_sec": float(elapsed),
            })
            pd.DataFrame(final_history).to_csv(
                final_dir / "final_history.csv", index=False, encoding="utf-8-sig"
            )
            print(
                "[final-all-seen] %s-%s epoch=%d/%d loss=%.4e"
                % (method, direction, epoch, best_epoch, float(averages["loss"])),
                flush=True,
            )
        if epoch % int(args.resume_save_every_epochs) == 0 or epoch == best_epoch:
            _v7_save_final_progress(
                cnn, final_resume, final_model, final_optimizer, final_scaler,
                epoch, final_history, final_rng, elapsed, model_config,
            )

    cnn.save_checkpoint(final_checkpoint, final_model, model_config, {
        "phase": "final_all_seen",
        "method": method,
        "direction": direction,
        "final_epochs": int(best_epoch),
        "seed": int(args.seed),
        "n_seen_pairs": int(store.n_seen),
        "all_seen_pairs_used": True,
        "selection_validation_pairs_reintroduced": True,
        "same_seen_unseen_csv_as_pinn": True,
        "intermediate_planes_used": 0,
    })
    cnn.write_json(final_dir / "train_config.json", {
        "script_version": SCRIPT_VERSION,
        "method": method,
        "direction": direction,
        "final_epochs": int(best_epoch),
        "n_final_training_pairs": int(len(all_seen_indices)),
        "all_seen_pairs_used": True,
        "learning_rate_schedule_replayed": True,
        "resume_supported": True,
        "model_config": model_config,
    })
    if final_resume.exists():
        final_resume.unlink()
    del final_model, store
    cnn.gc.collect()
    if device.type == "cuda":
        torch_module.cuda.empty_cache()
    return final_checkpoint


def _v7_ddnn_train_model(ddnn: Any, run_dir: Path, args: argparse.Namespace, device: Any) -> Path:
    np_module = ddnn.np
    torch_module = ddnn.torch
    seen = np_module.asarray(
        ddnn.load_combinations_csv(run_dir / "dataset" / "seen_combinations.csv"), dtype=np_module.float32
    )
    split_info = _v7_check_seen_unseen_split(ddnn, run_dir, int(seen.shape[1]))
    output_root = run_dir / str(args.output_folder)
    seed_root = output_root / "train" / ("seed_%d" % int(args.seed))
    selection_dir = seed_root / "model_selection"
    final_dir = seed_root / "final"
    final_checkpoint = final_dir / "final_ddnn.pt"
    selection_resume = selection_dir / "selection_resume.pt"
    final_resume = final_dir / "final_resume.pt"
    selection_summary_path = selection_dir / "selection_summary.json"
    best_checkpoint = selection_dir / "best_selection_ddnn.pt"

    if final_checkpoint.exists() and not bool(args.force_train):
        print("[train] exists, skip: %s" % final_checkpoint)
        return final_checkpoint
    if bool(args.force_train) and seed_root.exists():
        import shutil
        shutil.rmtree(seed_root)
    selection_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    store = ddnn.EndpointLabelStore(run_dir, str(args.source_cnn_folder))
    if len(seen) != store.n_seen:
        raise RuntimeError("Seen CSV count does not match endpoint-label dataset count.")
    model_config, pinn_checkpoint = ddnn.load_exact_model_config(run_dir)
    if int(model_config["n_pulses"]) != int(seen.shape[1]):
        raise RuntimeError("PINN checkpoint M does not match current run directory.")
    selection_train_ids, validation_ids = ddnn.make_config_split(
        len(seen), float(args.validation_fraction), int(args.split_seed)
    )
    all_seen_ids = np_module.arange(len(seen), dtype=np_module.int64)
    validation_data = ddnn.make_fixed_validation_points(
        store=store,
        seen_combinations=seen,
        validation_ids=validation_ids,
        points_per_endpoint=int(args.validation_points_per_endpoint),
        seed=int(args.split_seed) + 91000,
    )
    pd.DataFrame({
        "selection_training_index": pd.Series(selection_train_ids),
        "selection_validation_index": pd.Series(validation_ids),
        "final_all_seen_index": pd.Series(all_seen_ids),
    }).to_csv(selection_dir / "configuration_split.csv", index=False, encoding="utf-8-sig")
    ddnn.write_json(output_root / "experiment_manifest.json", {
        "script_version": SCRIPT_VERSION,
        "M": int(seen.shape[1]),
        "same_network_architecture_as_fourier_pinn": True,
        "pinn_architecture_checkpoint": str(pinn_checkpoint),
        "same_seen_unseen_csv_as_pinn": True,
        "split_integrity": split_info,
        "model_selection_training_configs": int(len(selection_train_ids)),
        "model_selection_validation_configs": int(len(validation_ids)),
        "final_training_configs": int(len(all_seen_ids)),
        "final_uses_every_seen_configuration": True,
        "supervision_planes": [0, int(store.shape[1] - 1)],
        "intermediate_planes_used": 0,
        "resume_supported": True,
        "physics_loss_used": False,
        "pde_residual_used": False,
        "convergence_required_before_final": False,
        "selection_accepts_best_checkpoint_at_budget": True,
    })

    amp_enabled = bool(device.type == "cuda" and not bool(args.no_amp))
    best_epoch = 0
    best_validation = float("inf")
    last_validation = float("nan")
    stopped_early = False
    history: List[Dict[str, Any]] = []
    lr_schedule: List[float] = []

    if selection_summary_path.exists() and not bool(args.force_train):
        summary = ddnn.read_json(selection_summary_path)
        if str(summary.get("convergence_status")) == "EARLY_STOP_CONVERGED" or bool(args.allow_unconverged):
            best_epoch = int(summary["best_epoch"])
            best_validation = float(summary["best_validation_loss"])
            stopped_early = bool(summary.get("stopped_early", False))
            lr_frame = pd.read_csv(selection_dir / "learning_rate_schedule.csv")
            lr_column = "learning_rate_used" if "learning_rate_used" in lr_frame.columns else "learning_rate"
            lr_schedule = [float(x) for x in lr_frame[lr_column].tolist()]
            print("[selection] DDNN completed previously; best_epoch=%d" % best_epoch)
        else:
            selection_summary_path.unlink()

    if best_epoch <= 0:
        ddnn.set_seed(int(args.seed))
        model = ddnn.ConditionalPINN(**model_config).to(device)
        optimizer = torch_module.optim.Adam(
            model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
        )
        scheduler = torch_module.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(args.lr_reduction_factor),
            patience=int(args.lr_patience_checks),
            min_lr=float(args.min_learning_rate),
        )
        scaler = torch_module.cuda.amp.GradScaler(enabled=amp_enabled)
        rng = np_module.random.default_rng(int(args.seed) + 123456)
        start_epoch = 1
        elapsed_before = 0.0
        if selection_resume.exists() and not bool(args.force_train):
            payload = _v7_torch_load(torch_module, selection_resume, device)
            model.load_state_dict(payload["model_state"])
            optimizer.load_state_dict(payload["optimizer_state"])
            scheduler.load_state_dict(payload["scheduler_state"])
            scaler.load_state_dict(payload.get("scaler_state", {}))
            best_epoch = int(payload.get("best_epoch", 0))
            best_validation = float(payload.get("best_loss", float("inf")))
            last_validation = float(payload.get("last_loss", float("nan")))
            history = list(payload.get("history", []))
            lr_schedule = [float(x) for x in payload.get("lr_schedule", [])]
            rng.bit_generator.state = payload["numpy_rng_state"]
            _v7_restore_torch_rng(torch_module, payload.get("torch_rng_state", {}))
            start_epoch = int(payload["epoch"]) + 1
            elapsed_before = float(payload.get("elapsed_sec", 0.0))
            print("[resume-selection] DDNN from epoch %d" % start_epoch)

        hard_max = int(args.hard_max_epochs)
        soft_max = int(args.max_epochs)
        extension = max(1, int(args.auto_extend_epochs))
        start_time = ddnn.time.perf_counter()
        stop_epoch = start_epoch - 1
        stale_epochs = 0 if best_epoch <= 0 else max(0, start_epoch - 1 - best_epoch)
        print(
            "[selection] DDNN params=%d train=%d val=%d soft_max=%d hard_max=%d patience=%d"
            % (
                ddnn.count_parameters(model), len(selection_train_ids), len(validation_ids),
                soft_max, hard_max, int(args.patience_epochs),
            )
        )
        for epoch in range(start_epoch, hard_max + 1):
            if epoch == soft_max + 1 or (
                epoch > soft_max + 1 and (epoch - soft_max - 1) % extension == 0
            ):
                print("[auto-extend] DDNN continues automatically: epoch %d, hard cap %d" % (epoch, hard_max))
            lr_used = float(optimizer.param_groups[0]["lr"])
            lr_schedule.append(lr_used)
            training = ddnn.train_one_epoch(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                store=store,
                seen_combinations=seen,
                config_ids=selection_train_ids,
                config_batch_size=int(args.config_batch_size),
                points_per_endpoint=int(args.points_per_endpoint),
                rng=rng,
                device=device,
                amp_enabled=amp_enabled,
                gradient_clip=float(args.gradient_clip),
                power_loss_weight=float(args.power_loss_weight),
            )
            should_validate = (
                epoch == 1
                or epoch % int(args.validate_every_epochs) == 0
                or epoch == hard_max
            )
            if not should_validate:
                continue
            validation = ddnn.validate_fixed_points(
                model=model,
                validation_data=validation_data,
                device=device,
                chunk_size=int(args.validation_chunk_size),
                amp_enabled=amp_enabled,
                power_loss_weight=float(args.power_loss_weight),
            )
            last_validation = float(validation["loss"])
            scheduler.step(last_validation)
            required = max(
                float(args.improvement_abs_tol),
                float(args.improvement_rel_tol) * abs(best_validation)
                if np_module.isfinite(best_validation) else 0.0,
            )
            improved = (
                not np_module.isfinite(best_validation)
                or last_validation < best_validation - required
            )
            if improved:
                best_validation = last_validation
                best_epoch = int(epoch)
                ddnn.save_checkpoint(
                    best_checkpoint,
                    model,
                    model_config,
                    {
                        "phase": "selection",
                        "best_epoch": best_epoch,
                        "best_validation_loss": best_validation,
                    },
                )
            stale_epochs = int(epoch - best_epoch) if best_epoch > 0 else int(epoch)
            elapsed = elapsed_before + float(ddnn.time.perf_counter() - start_time)
            history.append({
                "epoch": int(epoch),
                "train_loss": float(training["loss"]),
                "train_field_mse": float(training["field_mse"]),
                "train_power_mse": float(training["power_mse"]),
                "validation_loss": float(validation["loss"]),
                "validation_field_mse": float(validation["field_mse"]),
                "validation_power_mse": float(validation["power_mse"]),
                "learning_rate_used": float(lr_used),
                "learning_rate_next": float(optimizer.param_groups[0]["lr"]),
                "best_epoch": int(best_epoch),
                "stale_epochs": int(stale_epochs),
                "elapsed_sec": float(elapsed),
            })
            pd.DataFrame(history).to_csv(
                selection_dir / "selection_history.csv", index=False, encoding="utf-8-sig"
            )
            pd.DataFrame({
                "epoch": np_module.arange(1, len(lr_schedule) + 1, dtype=np_module.int64),
                "learning_rate_used": np_module.asarray(lr_schedule, dtype=np_module.float64),
            }).to_csv(
                selection_dir / "learning_rate_schedule.csv", index=False, encoding="utf-8-sig"
            )
            _v7_save_selection_progress(
                ddnn, selection_resume, model, optimizer, scheduler, scaler,
                epoch, best_epoch, best_validation, last_validation,
                history, lr_schedule, rng, elapsed, model_config,
            )
            print(
                "[select] DDNN epoch=%d train=%.4e val=%.4e best=%d stale=%d/%d lr=%.2e"
                % (
                    epoch, float(training["loss"]), last_validation, best_epoch,
                    stale_epochs, int(args.patience_epochs), lr_used,
                ),
                flush=True,
            )
            stop_epoch = int(epoch)
            if (
                epoch >= int(args.min_epochs)
                and best_epoch > 0
                and stale_epochs >= int(args.patience_epochs)
            ):
                stopped_early = True
                print("[select] DDNN early-stop convergence at epoch %d; best=%d" % (epoch, best_epoch))
                break

        status = "EARLY_STOP_CONVERGED" if stopped_early else "HARD_CAP_REACHED_NOT_ACCEPTED"
        ddnn.write_json(selection_summary_path, {
            "script_version": SCRIPT_VERSION,
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best_validation),
            "last_validation_loss": float(last_validation),
            "stop_epoch": int(stop_epoch),
            "stopped_early": bool(stopped_early),
            "convergence_status": status,
            "n_seen": int(len(seen)),
            "n_selection_training": int(len(selection_train_ids)),
            "n_selection_validation": int(len(validation_ids)),
            "n_final_training": int(len(all_seen_ids)),
            "final_uses_all_seen": True,
            "min_epochs": int(args.min_epochs),
            "soft_max_epochs": int(args.max_epochs),
            "hard_max_epochs": int(args.hard_max_epochs),
            "patience_epochs": int(args.patience_epochs),
            "model_config": model_config,
            "pinn_architecture_checkpoint": str(pinn_checkpoint),
        })
        if not stopped_early and not bool(args.allow_unconverged):
            raise RuntimeError(
                "DDNN M=%d reached hard cap %d without required early-stop convergence. "
                "Resume was kept and no final model was accepted."
                % (int(seen.shape[1]), int(args.hard_max_epochs))
            )

    if best_epoch <= 0:
        raise RuntimeError("No valid DDNN best epoch is available.")
    if len(lr_schedule) < best_epoch:
        lr_frame = pd.read_csv(selection_dir / "learning_rate_schedule.csv")
        lr_column = "learning_rate_used" if "learning_rate_used" in lr_frame.columns else "learning_rate"
        lr_schedule = [float(x) for x in lr_frame[lr_column].tolist()]
    if len(lr_schedule) < best_epoch:
        raise RuntimeError("DDNN learning-rate schedule is shorter than best_epoch.")

    ddnn.set_seed(int(args.seed))
    final_model = ddnn.ConditionalPINN(**model_config).to(device)
    final_optimizer = torch_module.optim.Adam(
        final_model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    final_scaler = torch_module.cuda.amp.GradScaler(enabled=amp_enabled)
    final_rng = np_module.random.default_rng(int(args.seed) + 654321)
    final_history: List[Dict[str, Any]] = []
    final_start_epoch = 1
    final_elapsed_before = 0.0
    if final_resume.exists() and not bool(args.force_train):
        payload = _v7_torch_load(torch_module, final_resume, device)
        final_model.load_state_dict(payload["model_state"])
        final_optimizer.load_state_dict(payload["optimizer_state"])
        final_scaler.load_state_dict(payload.get("scaler_state", {}))
        final_history = list(payload.get("history", []))
        final_rng.bit_generator.state = payload["numpy_rng_state"]
        _v7_restore_torch_rng(torch_module, payload.get("torch_rng_state", {}))
        final_start_epoch = int(payload["epoch"]) + 1
        final_elapsed_before = float(payload.get("elapsed_sec", 0.0))
        print("[resume-final] DDNN from epoch %d/%d" % (final_start_epoch, best_epoch))

    print(
        "[final-all-seen] DDNN epochs=%d all_seen_configs=%d batches/epoch=%d"
        % (
            best_epoch, len(all_seen_ids),
            int(math.ceil(len(all_seen_ids) / float(max(1, int(args.config_batch_size))))),
        )
    )
    final_start = ddnn.time.perf_counter()
    for epoch in range(final_start_epoch, best_epoch + 1):
        replay_lr = float(lr_schedule[epoch - 1])
        for group in final_optimizer.param_groups:
            group["lr"] = replay_lr
        training = ddnn.train_one_epoch(
            model=final_model,
            optimizer=final_optimizer,
            scaler=final_scaler,
            store=store,
            seen_combinations=seen,
            config_ids=all_seen_ids,
            config_batch_size=int(args.config_batch_size),
            points_per_endpoint=int(args.points_per_endpoint),
            rng=final_rng,
            device=device,
            amp_enabled=amp_enabled,
            gradient_clip=float(args.gradient_clip),
            power_loss_weight=float(args.power_loss_weight),
        )
        elapsed = final_elapsed_before + float(ddnn.time.perf_counter() - final_start)
        if epoch == 1 or epoch % int(args.log_every_epochs) == 0 or epoch == best_epoch:
            final_history.append({
                "epoch": int(epoch),
                "train_loss": float(training["loss"]),
                "train_field_mse": float(training["field_mse"]),
                "train_power_mse": float(training["power_mse"]),
                "learning_rate": float(replay_lr),
                "all_seen_configs_used": int(len(all_seen_ids)),
                "elapsed_sec": float(elapsed),
            })
            pd.DataFrame(final_history).to_csv(
                final_dir / "final_history.csv", index=False, encoding="utf-8-sig"
            )
            print(
                "[final-all-seen] DDNN epoch=%d/%d loss=%.4e"
                % (epoch, best_epoch, float(training["loss"])),
                flush=True,
            )
        if epoch % int(args.resume_save_every_epochs) == 0 or epoch == best_epoch:
            _v7_save_final_progress(
                ddnn, final_resume, final_model, final_optimizer, final_scaler,
                epoch, final_history, final_rng, elapsed, model_config,
            )

    ddnn.save_checkpoint(final_checkpoint, final_model, model_config, {
        "phase": "final_all_seen",
        "final_epochs": int(best_epoch),
        "seed": int(args.seed),
        "all_seen_configurations_used": True,
        "selection_validation_configurations_reintroduced": True,
        "same_seen_unseen_csv_as_pinn": True,
        "supervision": "z=0 and z=z_final only",
        "intermediate_planes_used": 0,
        "pinn_architecture_checkpoint": str(pinn_checkpoint),
    })
    ddnn.write_json(final_dir / "train_config.json", {
        "script_version": SCRIPT_VERSION,
        "final_epochs": int(best_epoch),
        "n_final_training_configurations": int(len(all_seen_ids)),
        "all_seen_configurations_used": True,
        "learning_rate_schedule_replayed": True,
        "resume_supported": True,
        "model_config": model_config,
    })
    if final_resume.exists():
        final_resume.unlink()
    del final_model, store
    ddnn.gc.collect()
    if device.type == "cuda":
        torch_module.cuda.empty_cache()
    return final_checkpoint


def _v7_ddnn_run_inverse(ddnn: Any, run_dir: Path, args: argparse.Namespace, device: Any) -> Dict[str, Any]:
    np_module = ddnn.np
    torch_module = ddnn.torch
    output_root = run_dir / str(args.output_folder)
    checkpoint = output_root / "train" / ("seed_%d" % int(args.seed)) / "final" / "final_ddnn.pt"
    if not checkpoint.exists():
        raise FileNotFoundError("Missing final DDNN checkpoint: %s" % checkpoint)
    inverse_root = output_root / "inverse" / ("seed_%d" % int(args.seed))
    result_path = inverse_root / "inverse_results.csv"
    summary_path = inverse_root / "summary.json"
    resume_path = inverse_root / "inverse_resume.pt"
    if result_path.exists() and summary_path.exists() and not bool(args.force_inverse):
        print("[inverse] exists, skip: %s" % result_path)
        return ddnn.read_json(summary_path)
    inverse_root.mkdir(parents=True, exist_ok=True)
    if bool(args.force_inverse):
        for path in (result_path, summary_path, resume_path):
            if path.exists():
                path.unlink()

    model, _ = ddnn.load_checkpoint(checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    unseen = np_module.asarray(
        ddnn.load_combinations_csv(run_dir / "dataset" / "unseen_combinations.csv"), dtype=np_module.float32
    )
    requested_inverse_samples = int(args.inverse_samples)
    n_samples = len(unseen) if requested_inverse_samples <= 0 else min(requested_inverse_samples, len(unseen))
    sample_rng = np_module.random.default_rng(
        int(args.inverse_sample_seed) + 1000 * int(unseen.shape[1])
    )
    selected_indices = np_module.sort(sample_rng.choice(len(unseen), size=n_samples, replace=False))
    true_amplitudes = unseen[selected_indices]

    target_batches: List[Any] = []
    tau = None
    z_final = None
    for start in range(0, n_samples, int(args.ssfm_batch_size)):
        part = true_amplitudes[start:start + int(args.ssfm_batch_size)]
        target_part, tau_part, z_part = ddnn.make_ssfm_terminal_batch(run_dir, part, device)
        target_batches.append(target_part)
        tau = tau_part
        z_final = z_part
    target_terminal = np_module.concatenate(target_batches, axis=0)
    if tau is None or z_final is None:
        raise RuntimeError("Failed to generate DDNN inverse targets.")

    sample_count, M = true_amplitudes.shape
    restarts = int(args.restarts)
    trajectory_count = sample_count * restarts
    sample_ids = np_module.repeat(np_module.arange(sample_count, dtype=np_module.int64), restarts)
    time_indices_np = ddnn.choose_time_indices(len(tau), int(args.terminal_points))
    tau_selected = np_module.asarray(tau[time_indices_np], dtype=np_module.float32)
    target_selected = torch_module.from_numpy(
        target_terminal[sample_ids][:, :, time_indices_np]
    ).to(device)

    raw = ddnn.initialize_raw(trajectory_count, M, int(args.seed) + 700000, device)
    optimizer = torch_module.optim.AdamW([raw], lr=float(args.inverse_learning_rate), weight_decay=0.0)
    scheduler = torch_module.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(args.inverse_epochs)),
        eta_min=float(args.inverse_min_learning_rate),
    )
    best_loss = torch_module.full((trajectory_count,), float("inf"), device=device)
    best_raw = raw.detach().clone()
    stale = torch_module.zeros((trajectory_count,), dtype=torch_module.long, device=device)
    start_epoch = 1
    elapsed_before = 0.0
    if resume_path.exists() and not bool(args.force_inverse):
        payload = _v7_torch_load(torch_module, resume_path, device)
        if payload.get("selected_indices") != selected_indices.tolist():
            raise RuntimeError("DDNN inverse resume sample indices do not match current settings.")
        raw.data.copy_(payload["raw"].to(device))
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        best_loss = payload["best_loss"].to(device)
        best_raw = payload["best_raw"].to(device)
        stale = payload["stale"].to(device)
        start_epoch = int(payload["epoch"]) + 1
        elapsed_before = float(payload.get("elapsed_sec", 0.0))
        _v7_restore_torch_rng(torch_module, payload.get("torch_rng_state", {}))
        print("[resume-inverse] DDNN from epoch %d/%d" % (start_epoch, int(args.inverse_epochs)))

    stopped_epoch = int(args.inverse_epochs)
    inverse_start = ddnn.time.perf_counter()
    plateau_fraction = 0.0
    print(
        "[inverse] samples=%d restarts=%d trajectories=%d terminal_points=%d resume=%s"
        % (sample_count, restarts, trajectory_count, len(time_indices_np), str(resume_path.exists()))
    )
    for epoch in range(start_epoch, int(args.inverse_epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        amplitudes = ddnn.raw_to_amplitudes(raw)
        prediction = ddnn.predict_terminal_waveforms(
            model=model,
            combinations=amplitudes,
            tau=tau_selected,
            z_final=float(z_final),
            device=device,
            chunk_size=int(args.inverse_point_chunk),
            amp_enabled=False,
            require_grad=True,
        )
        loss_vector = ddnn.relative_complex_squared_vector(
            prediction.float(), target_selected.float()
        )
        loss = torch_module.mean(loss_vector)
        if not torch_module.isfinite(loss):
            raise FloatingPointError("Non-finite DDNN inverse loss.")
        loss.backward()
        torch_module.nn.utils.clip_grad_norm_([raw], max_norm=10.0)
        optimizer.step()
        scheduler.step()
        with torch_module.no_grad():
            threshold = torch_module.maximum(
                torch_module.full_like(best_loss, float(args.inverse_early_stop_abs_delta)),
                float(args.inverse_early_stop_rel_delta) * torch_module.abs(best_loss),
            )
            improved = torch_module.isinf(best_loss) | (loss_vector < best_loss - threshold)
            best_loss = torch_module.where(improved, loss_vector, best_loss)
            best_raw = torch_module.where(improved[:, None], raw.detach(), best_raw)
            stale = torch_module.where(improved, torch_module.zeros_like(stale), stale + 1)
            plateau_fraction = float(
                torch_module.mean(
                    (stale >= int(args.inverse_early_stop_patience)).float()
                ).cpu()
            )
        elapsed = elapsed_before + float(ddnn.time.perf_counter() - inverse_start)
        if epoch % int(args.inverse_resume_every_epochs) == 0:
            _v7_atomic_torch_save(torch_module, {
                "kind": "ddnn_inverse_resume_v7",
                "epoch": int(epoch),
                "raw": raw.detach().cpu(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_loss": best_loss.detach().cpu(),
                "best_raw": best_raw.detach().cpu(),
                "stale": stale.detach().cpu(),
                "selected_indices": selected_indices.tolist(),
                "time_indices": time_indices_np.tolist(),
                "elapsed_sec": float(elapsed),
                "torch_rng_state": _v7_capture_torch_rng(torch_module),
            }, resume_path)
        if epoch == 1 or epoch % int(args.inverse_log_every) == 0 or epoch == int(args.inverse_epochs):
            print(
                "[inverse] epoch=%d/%d mean=%.4e best=%.4e plateau=%.1f%%"
                % (
                    epoch, int(args.inverse_epochs), float(loss.detach().cpu()),
                    float(torch_module.mean(best_loss).detach().cpu()),
                    100.0 * plateau_fraction,
                ),
                flush=True,
            )
        if (
            epoch >= int(args.inverse_early_stop_min_epochs)
            and plateau_fraction >= float(args.inverse_early_stop_fraction)
        ):
            stopped_epoch = int(epoch)
            print("[inverse] early stop at epoch %d" % epoch)
            break

    inverse_elapsed = elapsed_before + float(ddnn.time.perf_counter() - inverse_start)
    with torch_module.no_grad():
        all_amplitudes = ddnn.raw_to_amplitudes(best_raw).cpu().numpy()
        all_losses = best_loss.cpu().numpy()
    predicted_amplitudes = np_module.empty((sample_count, M), dtype=np_module.float32)
    selected_losses = np_module.empty(sample_count, dtype=np_module.float64)
    selected_restarts = np_module.empty(sample_count, dtype=np_module.int64)
    for sample_id in range(sample_count):
        positions = np_module.arange(sample_id * restarts, (sample_id + 1) * restarts, dtype=np_module.int64)
        best_position = int(positions[np_module.argmin(all_losses[positions])])
        predicted_amplitudes[sample_id] = all_amplitudes[best_position]
        selected_losses[sample_id] = float(all_losses[best_position])
        selected_restarts[sample_id] = int(best_position - sample_id * restarts)

    rounded = ddnn.nearest_pam4(predicted_amplitudes)
    true_initial = ddnn.make_initial_waveforms_numpy(true_amplitudes, tau)
    predicted_initial = ddnn.make_initial_waveforms_numpy(predicted_amplitudes, tau)
    initial_metrics = ddnn.batch_case_metrics(predicted_initial, true_initial)
    backcheck_batches: List[Any] = []
    for start in range(0, sample_count, int(args.ssfm_batch_size)):
        part = predicted_amplitudes[start:start + int(args.ssfm_batch_size)]
        back_part, _, _ = ddnn.make_ssfm_terminal_batch(run_dir, part, device)
        backcheck_batches.append(back_part)
    back_terminal = np_module.concatenate(backcheck_batches, axis=0)
    back_metrics = ddnn.batch_case_metrics(back_terminal, target_terminal)
    rows: List[Dict[str, Any]] = []
    for index in range(sample_count):
        amplitude_error = predicted_amplitudes[index] - true_amplitudes[index]
        rows.append({
            "sample_rank": int(index),
            "unseen_index": int(selected_indices[index]),
            "M": int(M),
            "true_amplitudes": ddnn.combo_text(true_amplitudes[index]),
            "predicted_amplitudes": ddnn.combo_text(predicted_amplitudes[index]),
            "rounded_amplitudes": ddnn.combo_text(rounded[index]),
            "amplitude_mae": float(np_module.mean(np_module.abs(amplitude_error))),
            "amplitude_rmse": float(np_module.sqrt(np_module.mean(amplitude_error ** 2))),
            "per_pulse_accuracy": float(np_module.mean(np_module.isclose(rounded[index], true_amplitudes[index], atol=1e-6))),
            "exact_match": int(bool(np_module.allclose(rounded[index], true_amplitudes[index], atol=1e-6))),
            "initial_rel_l2_field": float(initial_metrics["rel_l2_field"][index]),
            "initial_rel_l2_power": float(initial_metrics["rel_l2_power"][index]),
            "terminal_relative_squared_objective": float(selected_losses[index]),
            "ssfm_backcheck_rel_l2_field": float(back_metrics["rel_l2_field"][index]),
            "ssfm_backcheck_rel_l2_power": float(back_metrics["rel_l2_power"][index]),
            "best_restart": int(selected_restarts[index]),
            "stopped_epoch": int(stopped_epoch),
            "inverse_sec_per_sample": float(inverse_elapsed / max(sample_count, 1)),
        })
    pd.DataFrame(rows).to_csv(result_path, index=False, encoding="utf-8-sig")
    summary = ddnn.summarize_numeric_csv(result_path, summary_path)
    summary.update({
        "script_version": SCRIPT_VERSION,
        "checkpoint": str(checkpoint),
        "n_samples": int(sample_count),
        "selected_unseen_indices": selected_indices.tolist(),
        "same_unseen_csv_as_pinn": True,
        "restarts": int(restarts),
        "stopped_epoch": int(stopped_epoch),
        "inverse_principle": "freeze forward DDNN and optimize amplitude vector",
        "resume_supported": True,
    })
    ddnn.write_json(summary_path, summary)
    if resume_path.exists():
        resume_path.unlink()
    del model
    ddnn.gc.collect()
    if device.type == "cuda":
        torch_module.cuda.empty_cache()
    print("[inverse] saved: %s" % result_path)
    return summary


def _v8_cnn_train_one_model(
    cnn: Any,
    *,
    run_dir: Path,
    method: str,
    direction: str,
    args: argparse.Namespace,
    device: Any,
) -> Path:
    np_module = cnn.np
    torch_module = cnn.torch
    seen = np_module.asarray(
        cnn.load_combinations_csv(run_dir / "dataset" / "seen_combinations.csv"),
        dtype=np_module.float64,
    )
    unseen = np_module.asarray(
        cnn.load_combinations_csv(run_dir / "dataset" / "unseen_combinations.csv"),
        dtype=np_module.float64,
    )
    split_info = _v7_check_seen_unseen_split(cnn, run_dir, int(seen.shape[1]))
    store = cnn.EndpointPairStore(
        run_dir=run_dir,
        source_cnn_folder=str(args.source_cnn_folder),
        cache_in_ram=bool(args.cache_seen_in_ram),
    )
    if len(seen) != store.n_seen:
        raise RuntimeError("Seen CSV count does not match endpoint dataset count.")

    selection_train_indices, validation_indices = cnn.make_internal_split(
        n_seen=store.n_seen,
        validation_fraction=float(args.validation_fraction),
        split_seed=int(args.split_seed),
    )
    all_seen_indices = np_module.arange(store.n_seen, dtype=np_module.int64)

    experiment_root = run_dir / str(args.output_folder)
    model_root = experiment_root / ("%s_%s" % (method, direction))
    seed_root = model_root / "train" / ("seed_%d" % int(args.seed))
    selection_dir = seed_root / "model_selection"
    final_dir = seed_root / "final"
    final_checkpoint = final_dir / "final_model.pt"
    selection_resume = selection_dir / "selection_resume.pt"
    final_resume = final_dir / "final_resume.pt"
    selection_summary_path = selection_dir / "selection_summary.json"
    best_checkpoint = selection_dir / "best_selection_model.pt"

    if final_checkpoint.exists() and not bool(args.force_train):
        print("[train] exists, skip: %s" % final_checkpoint)
        del store
        return final_checkpoint
    if bool(args.force_train) and seed_root.exists():
        import shutil
        shutil.rmtree(seed_root)
    selection_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    cnn.write_json(model_root / "experiment_manifest.json", {
        "script_version": SCRIPT_VERSION,
        "M": int(seen.shape[1]),
        "run_dir": str(run_dir),
        "method": method,
        "direction": direction,
        "same_seen_unseen_csv_as_pinn": True,
        "split_integrity": split_info,
        "n_seen_pairs": int(len(seen)),
        "n_unseen_pairs": int(len(unseen)),
        "model_selection_only": {
            "training_pairs": int(len(selection_train_indices)),
            "validation_pairs": int(len(validation_indices)),
            "validation_fraction": float(args.validation_fraction),
            "split_seed": int(args.split_seed),
        },
        "final_training_pairs": int(len(all_seen_indices)),
        "final_uses_every_seen_pair": True,
        "unseen_used_for_model_selection": False,
        "training_planes_read": [0, int(store.n_planes_in_source_file - 1)],
        "intermediate_planes_used": 0,
        "resume_supported": True,
        "convergence_required_before_final": False,
        "selection_accepts_best_checkpoint_at_budget": True,
    })
    cnn.write_json(model_root / "split_indices.json", {
        "selection_training_indices": selection_train_indices.tolist(),
        "selection_validation_indices": validation_indices.tolist(),
        "final_training_indices": all_seen_indices.tolist(),
        "final_training_is_all_seen": True,
        "split_seed": int(args.split_seed),
        "unseen_indices_never_used_for_training_or_selection": True,
    })

    amp_enabled = bool(args.amp and device.type == "cuda")
    model_config: Dict[str, Any]
    best_epoch = 0
    best_validation_loss = float("inf")
    last_validation_loss = float("nan")
    stopped_early = False
    history: List[Dict[str, Any]] = []
    lr_schedule: List[float] = []

    if selection_summary_path.exists() and not bool(args.force_train):
        selection_summary = cnn.read_json(selection_summary_path)
        if str(selection_summary.get("convergence_status")) in ("EARLY_STOP_CONVERGED", "HARD_CAP_BEST_CHECKPOINT_ACCEPTED") or bool(args.allow_unconverged):
            best_epoch = int(selection_summary["best_epoch"])
            best_validation_loss = float(selection_summary["best_validation_loss"])
            stopped_early = bool(selection_summary.get("stopped_early", False))
            lr_frame = pd.read_csv(selection_dir / "learning_rate_schedule.csv")
            lr_column = "learning_rate_used" if "learning_rate_used" in lr_frame.columns else "learning_rate"
            lr_schedule = [float(x) for x in lr_frame[lr_column].tolist()]
            print("[selection] completed previously; best_epoch=%d" % best_epoch)
        else:
            selection_summary_path.unlink()

    if best_epoch <= 0:
        cnn.set_seed(int(args.seed))
        model, model_config = cnn.build_model(method, store.tau, args)
        model = model.to(device)
        optimizer = torch_module.optim.Adam(
            model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
        )
        scheduler = torch_module.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(args.lr_reduction_factor),
            patience=int(args.lr_patience_checks),
            min_lr=float(args.min_learning_rate),
        )
        scaler = torch_module.cuda.amp.GradScaler(enabled=amp_enabled)
        rng = np_module.random.default_rng(
            int(args.seed) + (100000 if method == "ddnn" else 0) + (10000 if direction == "inverse" else 0)
        )
        start_epoch = 1
        elapsed_before = 0.0

        if selection_resume.exists() and not bool(args.force_train):
            payload = _v7_torch_load(torch_module, selection_resume, device)
            model.load_state_dict(payload["model_state"])
            optimizer.load_state_dict(payload["optimizer_state"])
            scheduler.load_state_dict(payload["scheduler_state"])
            scheduler.patience = int(args.lr_patience_checks)
            scheduler.factor = float(args.lr_reduction_factor)
            scaler.load_state_dict(payload.get("scaler_state", {}))
            best_epoch = int(payload.get("best_epoch", 0))
            best_validation_loss = float(payload.get("best_loss", float("inf")))
            last_validation_loss = float(payload.get("last_loss", float("nan")))
            history = list(payload.get("history", []))
            lr_schedule = [float(x) for x in payload.get("lr_schedule", [])]
            rng.bit_generator.state = payload["numpy_rng_state"]
            _v7_restore_torch_rng(torch_module, payload.get("torch_rng_state", {}))
            start_epoch = int(payload["epoch"]) + 1
            elapsed_before = float(payload.get("elapsed_sec", 0.0))
            model_config = dict(payload.get("model_config", model_config))
            print("[resume-selection] %s-%s from epoch %d" % (method, direction, start_epoch))

        hard_max = int(args.hard_max_epochs)
        soft_max = int(args.max_epochs)
        extension = max(1, int(args.auto_extend_epochs))
        start_time = cnn.time.perf_counter()
        stop_epoch = start_epoch - 1
        print(
            "[selection] method=%s direction=%s params=%d train=%d val=%d "
            "soft_max=%d hard_max=%d patience=%d resume=%s"
            % (
                method, direction, cnn.count_parameters(model), len(selection_train_indices),
                len(validation_indices), soft_max, hard_max, int(args.patience_epochs),
                str(selection_resume.exists()),
            )
        )

        for epoch in range(start_epoch, hard_max + 1):
            if epoch == soft_max + 1 or (
                epoch > soft_max + 1 and (epoch - soft_max - 1) % extension == 0
            ):
                print(
                    "[auto-extend] %s-%s continues automatically: epoch %d, hard cap %d"
                    % (method, direction, epoch, hard_max),
                    flush=True,
                )
            lr_used = float(optimizer.param_groups[0]["lr"])
            lr_schedule.append(lr_used)
            model.train()
            totals = {"loss": 0.0, "field_rel_sq": 0.0, "power_rel_sq": 0.0}
            count = 0
            n_batches = 0
            for selected in cnn.iter_epoch_batches(
                selection_train_indices, int(args.batch_size), rng, shuffle=True
            ):
                source_np, target_np = store.batch(selected, direction)
                source = torch_module.from_numpy(source_np).to(device)
                target = torch_module.from_numpy(target_np).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch_module.cuda.amp.autocast(enabled=amp_enabled):
                    prediction = model(source)
                    parts = cnn.supervised_endpoint_loss(
                        prediction, target, float(args.field_loss_weight), float(args.power_loss_weight)
                    )
                if not bool(torch_module.isfinite(parts["loss"]).all().item()):
                    raise FloatingPointError("Non-finite CNN loss at epoch %d." % epoch)
                scaler.scale(parts["loss"]).backward()
                scaler.unscale_(optimizer)
                torch_module.nn.utils.clip_grad_norm_(model.parameters(), float(args.gradient_clip))
                scaler.step(optimizer)
                scaler.update()
                batch_count = int(len(selected))
                for key in totals:
                    totals[key] += float(parts[key].detach().cpu()) * batch_count
                count += batch_count
                n_batches += 1
                del source, target, prediction, parts
            average = {key: value / max(count, 1) for key, value in totals.items()}

            should_validate = (
                epoch == 1
                or epoch % int(args.validate_every_epochs) == 0
                or epoch == hard_max
            )
            if not should_validate:
                continue
            validation = cnn.validation_metrics(
                model=model,
                store=store,
                indices=validation_indices,
                direction=direction,
                device=device,
                batch_size=max(1, min(int(args.eval_batch_size), len(validation_indices))),
                field_weight=float(args.field_loss_weight),
                power_weight=float(args.power_loss_weight),
                amp_enabled=amp_enabled,
            )
            last_validation_loss = float(validation["loss"])
            if not np_module.isfinite(last_validation_loss):
                raise FloatingPointError("Non-finite CNN validation loss.")
            scheduler.step(last_validation_loss)
            required = max(
                float(args.improvement_abs_tol),
                float(args.improvement_rel_tol) * abs(best_validation_loss)
                if np_module.isfinite(best_validation_loss) else 0.0,
            )
            improved = (
                not np_module.isfinite(best_validation_loss)
                or last_validation_loss < best_validation_loss - required
            )
            if improved:
                best_validation_loss = last_validation_loss
                best_epoch = int(epoch)
                cnn.save_checkpoint(
                    best_checkpoint,
                    model,
                    model_config,
                    {
                        "phase": "selection",
                        "direction": direction,
                        "best_epoch": best_epoch,
                        "best_validation_loss": best_validation_loss,
                    },
                )
            stale_epochs = int(epoch - best_epoch) if best_epoch > 0 else int(epoch)
            elapsed = elapsed_before + float(cnn.time.perf_counter() - start_time)
            history.append({
                "epoch": int(epoch),
                "batches": int(n_batches),
                "train_pairs_seen": int(count),
                "train_loss": float(average["loss"]),
                "train_field_rel_sq": float(average["field_rel_sq"]),
                "train_power_rel_sq": float(average["power_rel_sq"]),
                "validation_loss": float(validation["loss"]),
                "validation_field_rel_sq": float(validation["field_rel_sq"]),
                "validation_power_rel_sq": float(validation["power_rel_sq"]),
                "best_epoch": int(best_epoch),
                "stale_epochs": int(stale_epochs),
                "learning_rate_used": float(lr_used),
                "learning_rate_next": float(optimizer.param_groups[0]["lr"]),
                "elapsed_sec": float(elapsed),
            })
            pd.DataFrame(history).to_csv(
                selection_dir / "selection_history.csv", index=False, encoding="utf-8-sig"
            )
            pd.DataFrame({
                "epoch": np_module.arange(1, len(lr_schedule) + 1, dtype=np_module.int64),
                "learning_rate_used": np_module.asarray(lr_schedule, dtype=np_module.float64),
            }).to_csv(
                selection_dir / "learning_rate_schedule.csv", index=False, encoding="utf-8-sig"
            )
            _v7_save_selection_progress(
                cnn, selection_resume, model, optimizer, scheduler, scaler,
                epoch, best_epoch, best_validation_loss, last_validation_loss,
                history, lr_schedule, rng, elapsed, model_config,
            )
            print(
                "[select] %s-%s epoch=%d train=%.4e val=%.4e best=%d stale=%d/%d lr=%.2e"
                % (
                    method, direction, epoch, float(average["loss"]), last_validation_loss,
                    best_epoch, stale_epochs, int(args.patience_epochs), lr_used,
                ),
                flush=True,
            )
            stop_epoch = int(epoch)
            if (
                epoch >= int(args.min_epochs)
                and best_epoch > 0
                and stale_epochs >= int(args.patience_epochs)
            ):
                stopped_early = True
                print("[select] early-stop convergence at epoch %d; best=%d" % (epoch, best_epoch))
                break

        convergence_status = "EARLY_STOP_CONVERGED" if stopped_early else "HARD_CAP_BEST_CHECKPOINT_ACCEPTED"
        cnn.write_json(selection_summary_path, {
            "script_version": SCRIPT_VERSION,
            "direction": direction,
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best_validation_loss),
            "last_validation_loss": float(last_validation_loss),
            "stop_epoch": int(stop_epoch),
            "stopped_early": bool(stopped_early),
            "convergence_status": convergence_status,
            "n_seen_pairs": int(store.n_seen),
            "n_selection_training_pairs": int(len(selection_train_indices)),
            "n_selection_validation_pairs": int(len(validation_indices)),
            "n_final_training_pairs": int(len(all_seen_indices)),
            "final_uses_all_seen_pairs": True,
            "min_epochs": int(args.min_epochs),
            "soft_max_epochs": int(args.max_epochs),
            "hard_max_epochs": int(args.hard_max_epochs),
            "patience_epochs": int(args.patience_epochs),
            "validate_every_epochs": int(args.validate_every_epochs),
            "parameter_count": int(cnn.count_parameters(model)),
            "model_config": model_config,
        })
        if best_epoch <= 0 or not best_checkpoint.exists():
            raise RuntimeError("CNN selection produced no valid best checkpoint.")
        if not stopped_early:
            print("[selection] hard cap reached; accepting the best validation checkpoint at epoch %d." % best_epoch, flush=True)
    else:
        model_config = dict(cnn.read_json(selection_summary_path)["model_config"])

    if best_epoch <= 0 or not best_checkpoint.exists():
        raise RuntimeError("No valid best-selection CNN checkpoint is available for all-seen fine-tuning.")

    batches_per_epoch = int(math.ceil(len(all_seen_indices) / float(max(1, int(args.batch_size)))))
    fine_tune_epochs = int(math.ceil(int(args.final_optimizer_updates) / float(max(1, batches_per_epoch))))
    fine_tune_epochs = max(int(args.final_min_epochs), fine_tune_epochs)
    fine_tune_epochs = min(int(args.final_max_epochs), fine_tune_epochs)
    fast_resume = final_dir / "final_fast_resume_v8.pt"

    cnn.set_seed(int(args.seed) + 700000)
    final_model, model_config = cnn.build_model(method, store.tau, args)
    final_model = final_model.to(device)
    final_optimizer = torch_module.optim.Adam(
        final_model.parameters(), lr=float(args.final_learning_rate), weight_decay=float(args.weight_decay)
    )
    final_scaler = torch_module.cuda.amp.GradScaler(enabled=amp_enabled)
    final_rng = np_module.random.default_rng(
        int(args.seed) + 700000 + (10000 if direction == "inverse" else 0)
    )
    final_history: List[Dict[str, Any]] = []
    final_start_epoch = 1
    final_elapsed_before = 0.0
    warm_start_source = "best_selection_checkpoint"

    if fast_resume.exists() and not bool(args.force_train):
        payload = _v7_torch_load(torch_module, fast_resume, device)
        final_model.load_state_dict(payload["model_state"])
        final_optimizer.load_state_dict(payload["optimizer_state"])
        final_scaler.load_state_dict(payload.get("scaler_state", {}))
        final_history = list(payload.get("history", []))
        final_rng.bit_generator.state = payload["numpy_rng_state"]
        _v7_restore_torch_rng(torch_module, payload.get("torch_rng_state", {}))
        final_start_epoch = int(payload["epoch"]) + 1
        final_elapsed_before = float(payload.get("elapsed_sec", 0.0))
        warm_start_source = str(payload.get("warm_start_source", "fast_resume_v8"))
        print("[resume-final-fast] %s-%s from fine-tune epoch %d/%d" % (
            method, direction, final_start_epoch, fine_tune_epochs
        ), flush=True)
    else:
        source_payload = None
        if final_resume.exists() and not bool(args.force_train):
            source_payload = _v7_torch_load(torch_module, final_resume, device)
            warm_start_source = "legacy_all_seen_resume_epoch_%d" % int(source_payload.get("epoch", 0))
            print("[adopt-legacy-final] %s-%s uses existing all-seen resume state at epoch %d" % (
                method, direction, int(source_payload.get("epoch", 0))
            ), flush=True)
        else:
            source_payload = _v7_torch_load(torch_module, best_checkpoint, device)
            warm_start_source = "best_selection_epoch_%d" % int(best_epoch)
            print("[warm-start-final] %s-%s from best selection epoch %d" % (
                method, direction, best_epoch
            ), flush=True)
        final_model.load_state_dict(source_payload["model_state"])

    print(
        "[final-all-seen-fast] method=%s direction=%s fine_tune_epochs=%d "
        "all_seen_pairs=%d batches/epoch=%d target_updates~%d lr=%.2e source=%s"
        % (
            method, direction, fine_tune_epochs, len(all_seen_indices), batches_per_epoch,
            fine_tune_epochs * batches_per_epoch, float(args.final_learning_rate), warm_start_source,
        ), flush=True
    )
    final_start_time = cnn.time.perf_counter()
    for epoch in range(final_start_epoch, fine_tune_epochs + 1):
        final_model.train()
        totals = {"loss": 0.0, "field_rel_sq": 0.0, "power_rel_sq": 0.0}
        count = 0
        n_batches = 0
        for selected in cnn.iter_epoch_batches(all_seen_indices, int(args.batch_size), final_rng, shuffle=True):
            source_np, target_np = store.batch(selected, direction)
            source = torch_module.from_numpy(source_np).to(device)
            target = torch_module.from_numpy(target_np).to(device)
            final_optimizer.zero_grad(set_to_none=True)
            with torch_module.cuda.amp.autocast(enabled=amp_enabled):
                prediction = final_model(source)
                parts = cnn.supervised_endpoint_loss(
                    prediction, target, float(args.field_loss_weight), float(args.power_loss_weight)
                )
            if not bool(torch_module.isfinite(parts["loss"]).all().item()):
                raise FloatingPointError("Non-finite final CNN fine-tuning loss at epoch %d." % epoch)
            final_scaler.scale(parts["loss"]).backward()
            final_scaler.unscale_(final_optimizer)
            torch_module.nn.utils.clip_grad_norm_(final_model.parameters(), float(args.gradient_clip))
            final_scaler.step(final_optimizer)
            final_scaler.update()
            batch_count = int(len(selected))
            for key in totals:
                totals[key] += float(parts[key].detach().cpu()) * batch_count
            count += batch_count
            n_batches += 1
            del source, target, prediction, parts
        averages = {key: value / max(count, 1) for key, value in totals.items()}
        elapsed = final_elapsed_before + float(cnn.time.perf_counter() - final_start_time)
        if epoch == 1 or epoch % int(args.log_every_epochs) == 0 or epoch == fine_tune_epochs:
            final_history.append({
                "fine_tune_epoch": int(epoch),
                "batches": int(n_batches),
                "all_seen_pairs_seen": int(count),
                "train_loss": float(averages["loss"]),
                "train_field_rel_sq": float(averages["field_rel_sq"]),
                "train_power_rel_sq": float(averages["power_rel_sq"]),
                "learning_rate": float(args.final_learning_rate),
                "elapsed_sec": float(elapsed),
                "warm_start_source": warm_start_source,
            })
            pd.DataFrame(final_history).to_csv(
                final_dir / "final_fast_history.csv", index=False, encoding="utf-8-sig"
            )
            print(
                "[final-all-seen-fast] %s-%s epoch=%d/%d loss=%.4e"
                % (method, direction, epoch, fine_tune_epochs, float(averages["loss"])),
                flush=True,
            )
        if epoch % int(args.resume_save_every_epochs) == 0 or epoch == fine_tune_epochs:
            _v7_save_final_progress(
                cnn, fast_resume, final_model, final_optimizer, final_scaler,
                epoch, final_history, final_rng, elapsed, model_config,
            )
            # Add V8 metadata without changing the portable resume payload format.
            payload = _v7_torch_load(torch_module, fast_resume, "cpu")
            payload["kind"] = "final_fast_resume_v8"
            payload["warm_start_source"] = warm_start_source
            payload["fine_tune_epochs_target"] = int(fine_tune_epochs)
            _v7_atomic_torch_save(torch_module, payload, fast_resume)

    cnn.save_checkpoint(final_checkpoint, final_model, model_config, {
        "phase": "final_all_seen_warmstart_finetune_v8",
        "method": method,
        "direction": direction,
        "selection_best_epoch": int(best_epoch),
        "selection_best_validation_loss": float(best_validation_loss),
        "fine_tune_epochs": int(fine_tune_epochs),
        "fine_tune_optimizer_updates_approx": int(fine_tune_epochs * batches_per_epoch),
        "fine_tune_learning_rate": float(args.final_learning_rate),
        "warm_start_source": warm_start_source,
        "seed": int(args.seed),
        "n_seen_pairs": int(store.n_seen),
        "all_seen_pairs_used": True,
        "selection_validation_pairs_reintroduced": True,
        "same_seen_unseen_csv_as_pinn": True,
        "intermediate_planes_used": 0,
    })
    cnn.write_json(final_dir / "train_config.json", {
        "script_version": SCRIPT_VERSION,
        "method": method,
        "direction": direction,
        "selection_best_epoch": int(best_epoch),
        "fine_tune_epochs": int(fine_tune_epochs),
        "n_final_training_pairs": int(len(all_seen_indices)),
        "all_seen_pairs_used": True,
        "final_strategy": "warm-start from selected/legacy model, then fine-tune on every seen pair",
        "warm_start_source": warm_start_source,
        "resume_supported": True,
        "model_config": model_config,
    })
    if fast_resume.exists():
        fast_resume.unlink()
    del final_model, store
    cnn.gc.collect()
    if device.type == "cuda":
        torch_module.cuda.empty_cache()
    return final_checkpoint




def _v8_ddnn_train_model(ddnn: Any, run_dir: Path, args: argparse.Namespace, device: Any) -> Path:
    np_module = ddnn.np
    torch_module = ddnn.torch
    seen = np_module.asarray(
        ddnn.load_combinations_csv(run_dir / "dataset" / "seen_combinations.csv"), dtype=np_module.float32
    )
    split_info = _v7_check_seen_unseen_split(ddnn, run_dir, int(seen.shape[1]))
    output_root = run_dir / str(args.output_folder)
    seed_root = output_root / "train" / ("seed_%d" % int(args.seed))
    selection_dir = seed_root / "model_selection"
    final_dir = seed_root / "final"
    final_checkpoint = final_dir / "final_ddnn.pt"
    selection_resume = selection_dir / "selection_resume.pt"
    final_resume = final_dir / "final_resume.pt"
    selection_summary_path = selection_dir / "selection_summary.json"
    best_checkpoint = selection_dir / "best_selection_ddnn.pt"

    if final_checkpoint.exists() and not bool(args.force_train):
        print("[train] exists, skip: %s" % final_checkpoint)
        return final_checkpoint
    if bool(args.force_train) and seed_root.exists():
        import shutil
        shutil.rmtree(seed_root)
    selection_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    store = ddnn.EndpointLabelStore(run_dir, str(args.source_cnn_folder))
    if len(seen) != store.n_seen:
        raise RuntimeError("Seen CSV count does not match endpoint-label dataset count.")
    model_config, pinn_checkpoint = ddnn.load_exact_model_config(run_dir)
    if int(model_config["n_pulses"]) != int(seen.shape[1]):
        raise RuntimeError("PINN checkpoint M does not match current run directory.")
    selection_train_ids, validation_ids = ddnn.make_config_split(
        len(seen), float(args.validation_fraction), int(args.split_seed)
    )
    all_seen_ids = np_module.arange(len(seen), dtype=np_module.int64)
    validation_data = ddnn.make_fixed_validation_points(
        store=store,
        seen_combinations=seen,
        validation_ids=validation_ids,
        points_per_endpoint=int(args.validation_points_per_endpoint),
        seed=int(args.split_seed) + 91000,
    )
    pd.DataFrame({
        "selection_training_index": pd.Series(selection_train_ids),
        "selection_validation_index": pd.Series(validation_ids),
        "final_all_seen_index": pd.Series(all_seen_ids),
    }).to_csv(selection_dir / "configuration_split.csv", index=False, encoding="utf-8-sig")
    ddnn.write_json(output_root / "experiment_manifest.json", {
        "script_version": SCRIPT_VERSION,
        "M": int(seen.shape[1]),
        "same_network_architecture_as_fourier_pinn": True,
        "pinn_architecture_checkpoint": str(pinn_checkpoint),
        "same_seen_unseen_csv_as_pinn": True,
        "split_integrity": split_info,
        "model_selection_training_configs": int(len(selection_train_ids)),
        "model_selection_validation_configs": int(len(validation_ids)),
        "final_training_configs": int(len(all_seen_ids)),
        "final_uses_every_seen_configuration": True,
        "supervision_planes": [0, int(store.shape[1] - 1)],
        "intermediate_planes_used": 0,
        "resume_supported": True,
        "physics_loss_used": False,
        "pde_residual_used": False,
        "convergence_required_before_final": False,
        "selection_accepts_best_checkpoint_at_budget": True,
    })

    amp_enabled = bool(device.type == "cuda" and not bool(args.no_amp))
    best_epoch = 0
    best_validation = float("inf")
    last_validation = float("nan")
    stopped_early = False
    history: List[Dict[str, Any]] = []
    lr_schedule: List[float] = []

    if selection_summary_path.exists() and not bool(args.force_train):
        summary = ddnn.read_json(selection_summary_path)
        if str(summary.get("convergence_status")) in ("EARLY_STOP_CONVERGED", "HARD_CAP_BEST_CHECKPOINT_ACCEPTED") or bool(args.allow_unconverged):
            best_epoch = int(summary["best_epoch"])
            best_validation = float(summary["best_validation_loss"])
            stopped_early = bool(summary.get("stopped_early", False))
            lr_frame = pd.read_csv(selection_dir / "learning_rate_schedule.csv")
            lr_column = "learning_rate_used" if "learning_rate_used" in lr_frame.columns else "learning_rate"
            lr_schedule = [float(x) for x in lr_frame[lr_column].tolist()]
            print("[selection] DDNN completed previously; best_epoch=%d" % best_epoch)
        else:
            selection_summary_path.unlink()

    if best_epoch <= 0:
        ddnn.set_seed(int(args.seed))
        model = ddnn.ConditionalPINN(**model_config).to(device)
        optimizer = torch_module.optim.Adam(
            model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
        )
        scheduler = torch_module.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(args.lr_reduction_factor),
            patience=int(args.lr_patience_checks),
            min_lr=float(args.min_learning_rate),
        )
        scaler = torch_module.cuda.amp.GradScaler(enabled=amp_enabled)
        rng = np_module.random.default_rng(int(args.seed) + 123456)
        start_epoch = 1
        elapsed_before = 0.0
        if selection_resume.exists() and not bool(args.force_train):
            payload = _v7_torch_load(torch_module, selection_resume, device)
            model.load_state_dict(payload["model_state"])
            optimizer.load_state_dict(payload["optimizer_state"])
            scheduler.load_state_dict(payload["scheduler_state"])
            scheduler.patience = int(args.lr_patience_checks)
            scheduler.factor = float(args.lr_reduction_factor)
            scaler.load_state_dict(payload.get("scaler_state", {}))
            best_epoch = int(payload.get("best_epoch", 0))
            best_validation = float(payload.get("best_loss", float("inf")))
            last_validation = float(payload.get("last_loss", float("nan")))
            history = list(payload.get("history", []))
            lr_schedule = [float(x) for x in payload.get("lr_schedule", [])]
            rng.bit_generator.state = payload["numpy_rng_state"]
            _v7_restore_torch_rng(torch_module, payload.get("torch_rng_state", {}))
            start_epoch = int(payload["epoch"]) + 1
            elapsed_before = float(payload.get("elapsed_sec", 0.0))
            print("[resume-selection] DDNN from epoch %d" % start_epoch)

        hard_max = int(args.hard_max_epochs)
        soft_max = int(args.max_epochs)
        extension = max(1, int(args.auto_extend_epochs))
        start_time = ddnn.time.perf_counter()
        stop_epoch = start_epoch - 1
        stale_epochs = 0 if best_epoch <= 0 else max(0, start_epoch - 1 - best_epoch)
        print(
            "[selection] DDNN params=%d train=%d val=%d soft_max=%d hard_max=%d patience=%d"
            % (
                ddnn.count_parameters(model), len(selection_train_ids), len(validation_ids),
                soft_max, hard_max, int(args.patience_epochs),
            )
        )
        for epoch in range(start_epoch, hard_max + 1):
            if epoch == soft_max + 1 or (
                epoch > soft_max + 1 and (epoch - soft_max - 1) % extension == 0
            ):
                print("[auto-extend] DDNN continues automatically: epoch %d, hard cap %d" % (epoch, hard_max))
            lr_used = float(optimizer.param_groups[0]["lr"])
            lr_schedule.append(lr_used)
            training = ddnn.train_one_epoch(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                store=store,
                seen_combinations=seen,
                config_ids=selection_train_ids,
                config_batch_size=int(args.config_batch_size),
                points_per_endpoint=int(args.points_per_endpoint),
                rng=rng,
                device=device,
                amp_enabled=amp_enabled,
                gradient_clip=float(args.gradient_clip),
                power_loss_weight=float(args.power_loss_weight),
            )
            should_validate = (
                epoch == 1
                or epoch % int(args.validate_every_epochs) == 0
                or epoch == hard_max
            )
            if not should_validate:
                continue
            validation = ddnn.validate_fixed_points(
                model=model,
                validation_data=validation_data,
                device=device,
                chunk_size=int(args.validation_chunk_size),
                amp_enabled=amp_enabled,
                power_loss_weight=float(args.power_loss_weight),
            )
            last_validation = float(validation["loss"])
            scheduler.step(last_validation)
            required = max(
                float(args.improvement_abs_tol),
                float(args.improvement_rel_tol) * abs(best_validation)
                if np_module.isfinite(best_validation) else 0.0,
            )
            improved = (
                not np_module.isfinite(best_validation)
                or last_validation < best_validation - required
            )
            if improved:
                best_validation = last_validation
                best_epoch = int(epoch)
                ddnn.save_checkpoint(
                    best_checkpoint,
                    model,
                    model_config,
                    {
                        "phase": "selection",
                        "best_epoch": best_epoch,
                        "best_validation_loss": best_validation,
                    },
                )
            stale_epochs = int(epoch - best_epoch) if best_epoch > 0 else int(epoch)
            elapsed = elapsed_before + float(ddnn.time.perf_counter() - start_time)
            history.append({
                "epoch": int(epoch),
                "train_loss": float(training["loss"]),
                "train_field_mse": float(training["field_mse"]),
                "train_power_mse": float(training["power_mse"]),
                "validation_loss": float(validation["loss"]),
                "validation_field_mse": float(validation["field_mse"]),
                "validation_power_mse": float(validation["power_mse"]),
                "learning_rate_used": float(lr_used),
                "learning_rate_next": float(optimizer.param_groups[0]["lr"]),
                "best_epoch": int(best_epoch),
                "stale_epochs": int(stale_epochs),
                "elapsed_sec": float(elapsed),
            })
            pd.DataFrame(history).to_csv(
                selection_dir / "selection_history.csv", index=False, encoding="utf-8-sig"
            )
            pd.DataFrame({
                "epoch": np_module.arange(1, len(lr_schedule) + 1, dtype=np_module.int64),
                "learning_rate_used": np_module.asarray(lr_schedule, dtype=np_module.float64),
            }).to_csv(
                selection_dir / "learning_rate_schedule.csv", index=False, encoding="utf-8-sig"
            )
            _v7_save_selection_progress(
                ddnn, selection_resume, model, optimizer, scheduler, scaler,
                epoch, best_epoch, best_validation, last_validation,
                history, lr_schedule, rng, elapsed, model_config,
            )
            print(
                "[select] DDNN epoch=%d train=%.4e val=%.4e best=%d stale=%d/%d lr=%.2e"
                % (
                    epoch, float(training["loss"]), last_validation, best_epoch,
                    stale_epochs, int(args.patience_epochs), lr_used,
                ),
                flush=True,
            )
            stop_epoch = int(epoch)
            if (
                epoch >= int(args.min_epochs)
                and best_epoch > 0
                and stale_epochs >= int(args.patience_epochs)
            ):
                stopped_early = True
                print("[select] DDNN early-stop convergence at epoch %d; best=%d" % (epoch, best_epoch))
                break

        status = "EARLY_STOP_CONVERGED" if stopped_early else "HARD_CAP_BEST_CHECKPOINT_ACCEPTED"
        ddnn.write_json(selection_summary_path, {
            "script_version": SCRIPT_VERSION,
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best_validation),
            "last_validation_loss": float(last_validation),
            "stop_epoch": int(stop_epoch),
            "stopped_early": bool(stopped_early),
            "convergence_status": status,
            "n_seen": int(len(seen)),
            "n_selection_training": int(len(selection_train_ids)),
            "n_selection_validation": int(len(validation_ids)),
            "n_final_training": int(len(all_seen_ids)),
            "final_uses_all_seen": True,
            "min_epochs": int(args.min_epochs),
            "soft_max_epochs": int(args.max_epochs),
            "hard_max_epochs": int(args.hard_max_epochs),
            "patience_epochs": int(args.patience_epochs),
            "model_config": model_config,
            "pinn_architecture_checkpoint": str(pinn_checkpoint),
        })
        if best_epoch <= 0 or not best_checkpoint.exists():
            raise RuntimeError("DDNN selection produced no valid best checkpoint.")
        if not stopped_early:
            print("[selection] DDNN hard cap reached; accepting best validation checkpoint at epoch %d." % best_epoch, flush=True)

    if best_epoch <= 0 or not best_checkpoint.exists():
        raise RuntimeError("No valid best-selection DDNN checkpoint is available for all-seen fine-tuning.")

    batches_per_epoch = int(math.ceil(len(all_seen_ids) / float(max(1, int(args.config_batch_size)))))
    fine_tune_epochs = int(math.ceil(int(args.final_optimizer_updates) / float(max(1, batches_per_epoch))))
    fine_tune_epochs = max(int(args.final_min_epochs), fine_tune_epochs)
    fine_tune_epochs = min(int(args.final_max_epochs), fine_tune_epochs)
    fast_resume = final_dir / "final_fast_resume_v8.pt"

    ddnn.set_seed(int(args.seed) + 800000)
    final_model = ddnn.ConditionalPINN(**model_config).to(device)
    final_optimizer = torch_module.optim.Adam(
        final_model.parameters(), lr=float(args.final_learning_rate), weight_decay=float(args.weight_decay)
    )
    final_scaler = torch_module.cuda.amp.GradScaler(enabled=amp_enabled)
    final_rng = np_module.random.default_rng(int(args.seed) + 800000)
    final_history: List[Dict[str, Any]] = []
    final_start_epoch = 1
    final_elapsed_before = 0.0
    warm_start_source = "best_selection_checkpoint"

    if fast_resume.exists() and not bool(args.force_train):
        payload = _v7_torch_load(torch_module, fast_resume, device)
        final_model.load_state_dict(payload["model_state"])
        final_optimizer.load_state_dict(payload["optimizer_state"])
        final_scaler.load_state_dict(payload.get("scaler_state", {}))
        final_history = list(payload.get("history", []))
        final_rng.bit_generator.state = payload["numpy_rng_state"]
        _v7_restore_torch_rng(torch_module, payload.get("torch_rng_state", {}))
        final_start_epoch = int(payload["epoch"]) + 1
        final_elapsed_before = float(payload.get("elapsed_sec", 0.0))
        warm_start_source = str(payload.get("warm_start_source", "fast_resume_v8"))
        print("[resume-final-fast] DDNN from fine-tune epoch %d/%d" % (
            final_start_epoch, fine_tune_epochs
        ), flush=True)
    else:
        source_payload = None
        if final_resume.exists() and not bool(args.force_train):
            source_payload = _v7_torch_load(torch_module, final_resume, device)
            warm_start_source = "legacy_all_seen_resume_epoch_%d" % int(source_payload.get("epoch", 0))
            print("[adopt-legacy-final] DDNN uses existing all-seen resume state at epoch %d" % (
                int(source_payload.get("epoch", 0))
            ), flush=True)
        else:
            source_payload = _v7_torch_load(torch_module, best_checkpoint, device)
            warm_start_source = "best_selection_epoch_%d" % int(best_epoch)
            print("[warm-start-final] DDNN from best selection epoch %d" % best_epoch, flush=True)
        final_model.load_state_dict(source_payload["model_state"])

    print(
        "[final-all-seen-fast] DDNN fine_tune_epochs=%d all_seen_configs=%d "
        "batches/epoch=%d target_updates~%d lr=%.2e source=%s"
        % (
            fine_tune_epochs, len(all_seen_ids), batches_per_epoch,
            fine_tune_epochs * batches_per_epoch, float(args.final_learning_rate), warm_start_source,
        ), flush=True
    )
    final_start_time = ddnn.time.perf_counter()
    for epoch in range(final_start_epoch, fine_tune_epochs + 1):
        training = ddnn.train_one_epoch(
            model=final_model,
            optimizer=final_optimizer,
            scaler=final_scaler,
            store=store,
            seen_combinations=seen,
            config_ids=all_seen_ids,
            config_batch_size=int(args.config_batch_size),
            points_per_endpoint=int(args.points_per_endpoint),
            rng=final_rng,
            device=device,
            amp_enabled=amp_enabled,
            gradient_clip=float(args.gradient_clip),
            power_loss_weight=float(args.power_loss_weight),
        )
        elapsed = final_elapsed_before + float(ddnn.time.perf_counter() - final_start_time)
        if epoch == 1 or epoch % int(args.log_every_epochs) == 0 or epoch == fine_tune_epochs:
            final_history.append({
                "fine_tune_epoch": int(epoch),
                "train_loss": float(training["loss"]),
                "train_field_mse": float(training["field_mse"]),
                "train_power_mse": float(training["power_mse"]),
                "learning_rate": float(args.final_learning_rate),
                "all_seen_configs_used": int(len(all_seen_ids)),
                "elapsed_sec": float(elapsed),
                "warm_start_source": warm_start_source,
            })
            pd.DataFrame(final_history).to_csv(
                final_dir / "final_fast_history.csv", index=False, encoding="utf-8-sig"
            )
            print(
                "[final-all-seen-fast] DDNN epoch=%d/%d loss=%.4e"
                % (epoch, fine_tune_epochs, float(training["loss"])),
                flush=True,
            )
        if epoch % int(args.resume_save_every_epochs) == 0 or epoch == fine_tune_epochs:
            _v7_save_final_progress(
                ddnn, fast_resume, final_model, final_optimizer, final_scaler,
                epoch, final_history, final_rng, elapsed, model_config,
            )
            payload = _v7_torch_load(torch_module, fast_resume, "cpu")
            payload["kind"] = "final_fast_resume_v8"
            payload["warm_start_source"] = warm_start_source
            payload["fine_tune_epochs_target"] = int(fine_tune_epochs)
            _v7_atomic_torch_save(torch_module, payload, fast_resume)

    ddnn.save_checkpoint(final_checkpoint, final_model, model_config, {
        "phase": "final_all_seen_warmstart_finetune_v8",
        "selection_best_epoch": int(best_epoch),
        "selection_best_validation_loss": float(best_validation),
        "fine_tune_epochs": int(fine_tune_epochs),
        "fine_tune_optimizer_updates_approx": int(fine_tune_epochs * batches_per_epoch),
        "fine_tune_learning_rate": float(args.final_learning_rate),
        "warm_start_source": warm_start_source,
        "seed": int(args.seed),
        "all_seen_configurations_used": True,
        "selection_validation_configurations_reintroduced": True,
        "same_seen_unseen_csv_as_pinn": True,
        "supervision": "z=0 and z=z_final only",
        "intermediate_planes_used": 0,
        "pinn_architecture_checkpoint": str(pinn_checkpoint),
    })
    ddnn.write_json(final_dir / "train_config.json", {
        "script_version": SCRIPT_VERSION,
        "selection_best_epoch": int(best_epoch),
        "fine_tune_epochs": int(fine_tune_epochs),
        "n_final_training_configurations": int(len(all_seen_ids)),
        "all_seen_configurations_used": True,
        "final_strategy": "warm-start from selected/legacy model, then fine-tune on every seen configuration",
        "warm_start_source": warm_start_source,
        "resume_supported": True,
        "model_config": model_config,
    })
    if fast_resume.exists():
        fast_resume.unlink()
    del final_model, store
    ddnn.gc.collect()
    if device.type == "cuda":
        torch_module.cuda.empty_cache()
    return final_checkpoint




def _install_v7_patches(cnn: Any, ddnn: Any) -> None:
    def cnn_train_wrapper(*, run_dir: Path, method: str, direction: str, args: argparse.Namespace, device: Any) -> Path:
        return _v8_cnn_train_one_model(
            cnn,
            run_dir=run_dir,
            method=method,
            direction=direction,
            args=args,
            device=device,
        )

    def ddnn_train_wrapper(run_dir: Path, args: argparse.Namespace, device: Any) -> Path:
        return _v8_ddnn_train_model(ddnn, run_dir, args, device)

    def ddnn_inverse_wrapper(run_dir: Path, args: argparse.Namespace, device: Any) -> Dict[str, Any]:
        return _v7_ddnn_run_inverse(ddnn, run_dir, args, device)

    cnn.train_one_model = cnn_train_wrapper
    ddnn.train_model = ddnn_train_wrapper
    ddnn.run_inverse = ddnn_inverse_wrapper


def main() -> None:
    print("[build] V10B_WINDOWS_CHECKPOINT_FIX_20260705", flush=True)
    args = build_parser().parse_args()
    _validate_args(args)
    root = Path(args.runs_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError("Runs root does not exist: %s" % root)

    cnn = _load_embedded_module("_endpoint_cnn_impl", _CNN_SOURCE_B85)
    ddnn = _load_embedded_module("_samearch_ddnn_impl", _DDNN_SOURCE_B85)
    _install_v7_patches(cnn, ddnn)
    # Keep every embedded helper consistent with the explicit largest-ratio rule.
    cnn.base.find_run_dir = _find_run_dir_highest_ratio
    ddnn.find_run_dir = _find_run_dir_highest_ratio
    device = cnn.safe_device(args.device)
    m_values = sorted(set(int(value) for value in args.M))
    methods = tuple(dict.fromkeys(str(value).lower() for value in args.methods))
    failures: List[Dict[str, Any]] = []
    selected_run_dirs = {
        str(M): str(_find_run_dir_highest_ratio(root, int(M)))
        for M in m_values
    }

    print("=" * 96)
    print("SCRIPT VERSION: %s" % SCRIPT_VERSION)
    print("METHODS: %s" % list(methods))
    print("CNN FORWARD: h(0,t) -> h(4LD,t), independently trained")
    print("CNN INVERSE: h(4LD,t) -> h(0,t), independently trained")
    print("DDNN FORWARD: exact Fourier-PINN ConditionalPINN architecture")
    print("DDNN LABELS: z=0 and z=4LD only; no PDE and no middle planes")
    print("DDNN TIME SAMPLING: random training coordinates each epoch; complete endpoint grid for validation by default")
    print("DDNN INVERSE: freeze forward DDNN and optimize amplitudes (fair PINN comparison)")
    print("EVALUATION: SSFM and model inference are batched")
    print("FORMAL DATA RULE: selection uses 80%/20% of seen only to choose best_epoch")
    print("FINAL DATA RULE: warm-start from the selected checkpoint and fine-tune on 100% of original seen waveforms")
    print("TEST DATA RULE: evaluate the unchanged original unseen CSV; unseen never selects epochs")
    print("RESUME: selection, fast all-seen fine-tuning, evaluation, and DDNN inverse support restart")
    print(
        "SELECTION RULE: early stop when stable; otherwise stop at the M-specific "
        "optimizer-update budget and accept the historical best validation checkpoint."
    )
    print(
        "CNN BUDGET: max_updates=%d, min_epochs=%d | DDNN BUDGET: max_updates=%d, min_epochs=%d"
        % (
            int(args.cnn_max_optimizer_updates), int(args.cnn_min_epochs),
            int(args.ddnn_max_optimizer_updates), int(args.ddnn_min_epochs),
        )
    )
    print("DEVICE: %s | AMP: %s" % (device, str(bool(args.amp and device.type == "cuda"))))
    print("=" * 96)

    if "cnn" in methods:
        _run_cnn(cnn, args, device, m_values, failures)
    if "ddnn" in methods:
        _run_ddnn(ddnn, args, device, m_values, failures)

    summary_path = _collect_unified_summary(root, args.output_folder, m_values, args.seed)
    report_path = root / (args.output_folder + "_workflow_report.json")
    _write_json(
        report_path,
        {
            "script_version": SCRIPT_VERSION,
            "methods": list(methods),
            "workflow": args.workflow,
            "M_values": m_values,
            "selected_run_dirs": selected_run_dirs,
            "run_selection_rule": "largest parsed _r ratio for each M",
            "ddnn_time_sampling_rule": (
                "training randomly resamples --ddnn-points-per-endpoint unique time coordinates "
                "per endpoint/config/epoch; validation uses the complete stored time grid when "
                "--ddnn-validation-points-per-endpoint=0"
            ),
            "train_label_fraction": 1.0,
            "final_training_uses_all_seen": True,
            "final_strategy": "warm-start the selected best checkpoint and fine-tune on all seen",
            "cnn_final_optimizer_updates": int(args.cnn_final_optimizer_updates),
            "ddnn_final_optimizer_updates": int(args.ddnn_final_optimizer_updates),
            "model_selection_uses_internal_validation_only": True,
            "label_subset_seed": int(args.label_subset_seed),
            "validation_fraction": float(args.validation_fraction),
            "cnn_auto_epoch_by_updates": bool(args.cnn_auto_epoch_by_updates),
            "cnn_min_optimizer_updates": int(args.cnn_min_optimizer_updates),
            "cnn_max_optimizer_updates": int(args.cnn_max_optimizer_updates),
            "cnn_patience_optimizer_updates": int(args.cnn_patience_optimizer_updates),
            "cnn_lr_patience_optimizer_updates": int(args.cnn_lr_patience_optimizer_updates),
            "cnn_hard_max_epochs": int(args.cnn_hard_max_epochs),
            "ddnn_auto_epoch_by_updates": bool(args.ddnn_auto_epoch_by_updates),
            "ddnn_min_optimizer_updates": int(args.ddnn_min_optimizer_updates),
            "ddnn_max_optimizer_updates": int(args.ddnn_max_optimizer_updates),
            "ddnn_patience_optimizer_updates": int(args.ddnn_patience_optimizer_updates),
            "ddnn_lr_patience_optimizer_updates": int(args.ddnn_lr_patience_optimizer_updates),
            "ddnn_hard_max_epochs": int(args.ddnn_hard_max_epochs),
            "selection_acceptance_rule": "early stop or best validation checkpoint at update budget",
            "auto_extend_epochs": int(args.auto_extend_epochs),
            "resume_save_every_epochs": int(args.resume_save_every_epochs),
            "accept_unconverged": bool(args.allow_unconverged),
            "cnn_inverse_type": "independent direct waveform network",
            "ddnn_inverse_type": "frozen forward model plus amplitude optimization",
            "unified_summary": str(summary_path),
            "failures": failures,
        },
    )

    gc.collect()
    if device.type == "cuda":
        import torch
        torch.cuda.empty_cache()
    print("\n" + "=" * 96)
    print("WORKFLOW FINISHED")
    print("SUMMARY: %s" % summary_path)
    print("REPORT: %s" % report_path)
    print("FAILURES: %d" % len(failures))
    print("=" * 96)


if __name__ == "__main__":
    main()

