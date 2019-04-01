# Load Radon data
> radon <- read.csv('radon.csv')
> names(radon)
[1] "state"    "region"   "typebldg" "floor"    "room"     "wave"    
[7] "rep"      "radon"    "county"  
> dim(radon)
[1] 12777     9
> z <- lm(radon ~ county + floor + typebldg , data = radon)
> summary(z)

Call:
lm(formula = radon ~ county + floor + typebldg, data = radon)

Residuals:
    Min      1Q  Median      3Q     Max 
-4.8853 -0.5818  0.0120  0.5980  4.9741 

Coefficients:
                            Estimate Std. Error t value Pr(>|t|)    
(Intercept)                 1.555239   0.208025   7.476 8.16e-14 ***
countyADAIR                -0.701062   0.455642  -1.539 0.123922    
countyADAMS                -0.053314   0.242150  -0.220 0.825743    
countyAITKIN               -0.704999   0.538455  -1.309 0.190458    
countyALLEGHENY            -0.477014   0.216331  -2.205 0.027471 *  
countyALLEN                -0.714868   0.220922  -3.236 0.001216 ** 
countyANDREW               -0.376171   0.455653  -0.826 0.409066    
countyANOKA                -0.553931   0.248917  -2.225 0.026075 *  
countyAPACHE               -1.279393   0.330213  -3.874 0.000107 ***
countyARMSTRONG            -0.036274   0.336932  -0.108 0.914266    
countyATCHISON              0.745330   0.538465   1.384 0.166330    
countyAUDRAIN              -1.848293   0.455731  -4.056 5.03e-05 ***
countyBAD RIVER            -1.515663   0.407976  -3.715 0.000204 ***
countyBARNES                0.368073   0.262595   1.402 0.161036    
countyBARNSTABLE           -0.993532   0.230519  -4.310 1.65e-05 ***
countyBARRY                -0.946617   0.376491  -2.514 0.011939 *  
countyBARTHOLOMEW          -0.061931   0.279706  -0.221 0.824774    
countyBARTON               -1.477536   0.376589  -3.923 8.78e-05 ***
countyBATES                -0.662655   0.364368  -1.819 0.068990 .  
countyBAY MILLS            -1.740983   0.317923  -5.476 4.43e-08 ***
countyBEAVER                0.064046   0.226412   0.283 0.777278    
countyBECKER               -0.276791   0.610132  -0.454 0.650084    
countyBEDFORD               0.295805   0.336932   0.878 0.379996    
countyBELTRAMI             -0.193349   0.429051  -0.451 0.652255    
countyBENSON                0.021397   0.407992   0.052 0.958175    
countyBENTON               -1.072567   0.344887  -3.110 0.001876 ** 
countyBERKS                 0.230512   0.260115   0.886 0.375533    
countyBERKSHIRE            -0.742418   0.252943  -2.935 0.003340 ** 
countyBIG STONE             0.123439   0.610138   0.202 0.839675    
countyBILLINGS              0.499747   0.390804   1.279 0.201004    
countyBLACKFORD            -1.462718   0.538467  -2.716 0.006608 ** 
countyBLAIR                -0.504725   0.271727  -1.857 0.063268 .  
countyBLUE EARTH            0.545298   0.336935   1.618 0.105601    
countyBOIS FORTE           -0.672850   0.296411  -2.270 0.023225 *  
countyBOLLINGER            -1.127608   0.376492  -2.995 0.002750 ** 
countyBOONE                -0.438888   0.296445  -1.481 0.138764    
countyBOTTINEAU             0.043509   0.270009   0.161 0.871985    
countyBOWMAN                0.323702   0.273543   1.183 0.236685    
countyBRADFORD              0.038304   0.260108   0.147 0.882929    
countyBRISTOL              -0.817994   0.227200  -3.600 0.000319 ***
countyBROWN                 0.140396   0.429051   0.327 0.743504    
countyBUCHANAN             -0.230862   0.261318  -0.883 0.377008    
countyBUCKS                -0.306658   0.253836  -1.208 0.227033    
countyBURKE                -0.505071   0.538463  -0.938 0.348269    
countyBURLEIGH             -0.001913   0.229656  -0.008 0.993354    
countyBUTLER               -0.097793   0.228084  -0.429 0.668107    
countyCALDWELL             -0.634332   0.610130  -1.040 0.298515    
countyCALLAWAY             -0.999777   0.610138  -1.639 0.101320    
countyCAMBRIA              -0.493426   0.257832  -1.914 0.055675 .  
countyCAMDEN               -0.753941   0.290034  -2.599 0.009347 ** 
countyCAMERON              -0.555660   0.490451  -1.133 0.257254    
countyCAPE GIRARDEAU       -0.512986   0.376494  -1.363 0.173055    
countyCARBON                0.210724   0.312860   0.674 0.500615    
countyCARLTON              -0.448922   0.376498  -1.192 0.233143    
countyCARROLL              -0.803816   0.364377  -2.206 0.027402 *  
countyCARTER               -0.445359   0.732742  -0.608 0.543332    
countyCARVER               -0.253127   0.455638  -0.556 0.578532    
countyCASS                  0.032435   0.217008   0.149 0.881188    
countyCAVALIER             -0.861216   0.336949  -2.556 0.010602 *  
countyCEDAR                -1.497103   0.407979  -3.670 0.000244 ***
countyCENTRE                0.779662   0.296424   2.630 0.008543 ** 
countyCHARITON             -1.057676   0.538458  -1.964 0.049521 *  
countyCHESTER              -0.101145   0.269991  -0.375 0.707948    
countyCHIPPEWA              0.340676   0.538463   0.633 0.526953    
countyCHISAGO              -0.350850   0.455649  -0.770 0.441315    
countyCHRISTIAN            -1.432323   0.429059  -3.338 0.000845 ***
countyCLARION              -0.426616   0.344896  -1.237 0.216133    
countyCLARK                -0.711148   0.230529  -3.085 0.002041 ** 
countyCLAY                 -0.409324   0.226872  -1.804 0.071223 .  
countyCLEARFIELD           -0.285285   0.279706  -1.020 0.307775    
countyCLEARWATER           -0.280703   0.538538  -0.521 0.602214    
countyCLINTON              -0.596376   0.312797  -1.907 0.056597 .  
countyCOCHISE              -1.277813   0.261611  -4.884 1.05e-06 ***
countyCOCONINO             -1.302659   0.232651  -5.599 2.20e-08 ***
countyCOLE                 -0.896952   0.268358  -3.342 0.000833 ***
countyCOLUMBIA              0.666516   0.376498   1.770 0.076701 .  
countyCOOK                 -0.724709   0.732749  -0.989 0.322669    
countyCOOPER               -0.628988   0.538481  -1.168 0.242798    
countyCOTTONWOOD           -0.755967   0.538458  -1.404 0.160359    
countyCRAWFORD             -0.900787   0.275476  -3.270 0.001079 ** 
countyCROW WING            -0.423992   0.353951  -1.198 0.230986    
countyCUMBERLAND            1.009348   0.254768   3.962 7.48e-05 ***
countyDADE                 -0.299655   1.015324  -0.295 0.767897    
countyDAKOTA               -0.080356   0.242164  -0.332 0.740028    
countyDALLAS               -1.317474   0.538458  -2.447 0.014429 *  
countyDAUPHIN               0.660092   0.251273   2.627 0.008625 ** 
countyDAVIESS              -0.816877   0.353949  -2.308 0.021021 *  
countyDE KALB              -0.289177   0.287176  -1.007 0.313969    
countyDEARBORN             -0.777071   0.455639  -1.705 0.088135 .  
countyDECATUR              -0.245462   0.490451  -0.500 0.616745    
countyDELAWARE             -0.930740   0.247489  -3.761 0.000170 ***
countyDENT                 -2.069074   0.455640  -4.541 5.65e-06 ***
countyDICKEY                0.219670   0.364387   0.603 0.546621    
countyDIVIDE                0.631258   0.538463   1.172 0.241085    
countyDODGE                 0.410744   0.610138   0.673 0.500833    
countyDOUGLAS              -0.324153   0.317908  -1.020 0.307918    
countyDUBOIS               -1.224868   0.490448  -2.497 0.012522 *  
countyDUKES                -0.869228   0.456004  -1.906 0.056649 .  
countyDUNKLIN              -1.199518   0.353964  -3.389 0.000704 ***
countyDUNN                  0.479302   0.279720   1.714 0.086645 .  
countyEDDY                 -0.545780   0.538455  -1.014 0.310792    
countyELK                  -0.489694   0.317923  -1.540 0.123516    
countyELKHART              -0.273907   0.236542  -1.158 0.246902    
countyEMMONS                0.181538   0.329881   0.550 0.582114    
countyERIE                 -0.764648   0.238896  -3.201 0.001374 ** 
countyESSEX                -0.441725   0.218806  -2.019 0.043530 *  
countyFARIBAULT            -0.856759   0.455642  -1.880 0.060086 .  
countyFAYETTE              -0.328041   0.268353  -1.222 0.221571    
countyFILLMORE             -0.301518   0.732742  -0.411 0.680718    
countyFLOYD                -0.627099   0.271712  -2.308 0.021018 *  
countyFOND DU LAC          -0.078822   0.336943  -0.234 0.815041    
countyFOREST               -0.930372   0.610130  -1.525 0.127316    
countyFOREST COUNTY        -1.660929   0.455638  -3.645 0.000268 ***
countyFOSTER               -0.307912   0.429054  -0.718 0.472984    
countyFOUNTAIN              0.135948   0.344887   0.394 0.693453    
countyFRANKLIN             -0.615382   0.231992  -2.653 0.007998 ** 
countyFREEBORN              0.591820   0.390806   1.514 0.129960    
countyFULTON               -0.006939   0.353953  -0.020 0.984359    
countyGASCONADE            -2.244026   0.429050  -5.230 1.72e-07 ***
countyGENTRY               -0.864099   0.490445  -1.762 0.078117 .  
countyGIBSON               -0.740024   0.323571  -2.287 0.022209 *  
countyGILA                 -1.483245   0.344903  -4.300 1.72e-05 ***
countyGOLDEN VALLEY        -0.121061   0.429062  -0.282 0.777832    
countyGOODHUE               0.472027   0.336932   1.401 0.161253    
countyGRAHAM               -1.637953   0.277678  -5.899 3.76e-09 ***
countyGRAND FORKS           0.852564   0.220686   3.863 0.000112 ***
countyGRAND PORTAGE        -0.508475   0.287177  -1.771 0.076652 .  
countyGRAND TRAVERSE       -1.569784   0.329907  -4.758 1.97e-06 ***
countyGRANT                 0.259673   0.265326   0.979 0.327751    
countyGREENE               -0.804843   0.241148  -3.338 0.000848 ***
countyGREENLEE             -1.384914   0.407992  -3.394 0.000690 ***
countyGRIGGS               -0.322103   0.376492  -0.856 0.392269    
countyGRUNDY               -1.185179   0.538465  -2.201 0.027752 *  
countyHAMILTON             -0.491359   0.293102  -1.676 0.093683 .  
countyHAMPDEN              -1.027208   0.225540  -4.554 5.30e-06 ***
countyHAMPSHIRE            -0.731660   0.247516  -2.956 0.003122 ** 
countyHANCOCK              -0.603954   0.407979  -1.480 0.138803    
countyHANNAHVILLE          -1.770139   0.407974  -4.339 1.44e-05 ***
countyHARRISON             -0.223198   0.287184  -0.777 0.437059    
countyHENDRICKS            -1.026022   0.296416  -3.461 0.000539 ***
countyHENNEPIN             -0.072259   0.228853  -0.316 0.752201    
countyHENRY                -0.954269   0.261316  -3.652 0.000262 ***
countyHETTINGER             0.293444   0.273551   1.073 0.283418    
countyHICKORY              -1.611889   0.733620  -2.197 0.028027 *  
countyHOLT                 -1.014928   0.733620  -1.383 0.166551    
countyHOUSTON               0.175829   0.455639   0.386 0.699580    
countyHOWARD               -0.591851   0.271777  -2.178 0.029447 *  
countyHOWELL               -0.487028   0.329876  -1.476 0.139863    
countyHUBBARD              -0.522307   0.490446  -1.065 0.286914    
countyHUNTINGDON           -0.147732   0.429064  -0.344 0.730618    
countyHUNTINGTON           -0.470831   0.344889  -1.365 0.172225    
countyINDIANA              -0.340146   0.317921  -1.070 0.284682    
countyIRON                 -1.788043   0.429076  -4.167 3.10e-05 ***
countyISANTI               -0.333570   0.610138  -0.547 0.584586    
countyITASCA               -0.463812   0.364382  -1.273 0.203086    
countyJACKSON              -0.388107   0.215513  -1.801 0.071750 .  
countyJASPER               -1.284880   0.249655  -5.147 2.69e-07 ***
countyJAY                  -0.556679   0.490446  -1.135 0.256378    
countyJEFFERSON            -0.664443   0.235498  -2.821 0.004789 ** 
countyJENNINGS             -0.939868   0.308229  -3.049 0.002299 ** 
countyJOHNSON              -0.833897   0.240670  -3.465 0.000532 ***
countyJUNIATA              -0.557725   0.490445  -1.137 0.255485    
countyKANABEC              -0.153281   0.538463  -0.285 0.775907    
countyKANDIYOHI             0.672303   0.538463   1.249 0.211850    
countyKEWEENAW BAY         -1.570799   0.243767  -6.444 1.21e-10 ***
countyKIDDER               -0.160587   0.407984  -0.394 0.693875    
countyKITTSON              -0.215049   0.610132  -0.352 0.724497    
countyKNOX                 -0.809466   0.353987  -2.287 0.022230 *  
countyKOOCHICHING          -0.926873   0.429051  -2.160 0.030770 *  
countyKOSCIUSKO            -0.022114   0.275471  -0.080 0.936017    
countyLA MOURE              0.215658   0.490455   0.440 0.660154    
countyLA PAZ               -3.137347   0.732752  -4.282 1.87e-05 ***
countyLA PORTE             -0.533011   0.240675  -2.215 0.026802 *  
countyLAC COURTE OREILLES  -0.822248   0.244937  -3.357 0.000790 ***
countyLAC DU FLAMBEAU      -0.369957   0.265321  -1.394 0.163229    
countyLAC QUI PARLE         1.257457   0.732742   1.716 0.086169 .  
countyLAC VIEUX DESERT     -1.165350   0.353955  -3.292 0.000996 ***
countyLACKAWANNA           -0.779350   0.235510  -3.309 0.000938 ***
countyLACLEDE              -1.417951   0.353955  -4.006 6.21e-05 ***
countyLAFAYETTE            -0.347114   0.268348  -1.294 0.195856    
countyLAGRANGE              0.434975   0.390936   1.113 0.265880    
countyLAKE                 -1.366951   0.224336  -6.093 1.14e-09 ***
countyLAKE OF THE WOODS     0.166207   0.538454   0.309 0.757575    
countyLANCASTER             0.877442   0.239327   3.666 0.000247 ***
countyLAWRENCE             -0.626976   0.230983  -2.714 0.006649 ** 
countyLE SUEUR              0.233720   0.490448   0.477 0.633696    
countyLEBANON               1.181597   0.300027   3.938 8.25e-05 ***
countyLEECH LAKE           -0.925415   0.234557  -3.945 8.01e-05 ***
countyLEHIGH                0.830494   0.293102   2.833 0.004612 ** 
countyLEWIS                -2.305862   1.015322  -2.271 0.023160 *  
countyLINCOLN              -0.596905   0.376492  -1.585 0.112892    
countyLINN                 -0.645043   0.490462  -1.315 0.188476    
countyLIVINGSTON           -0.281057   0.732749  -0.384 0.701307    
countyLOGAN                 0.223527   0.344897   0.648 0.516933    
countyLOWER SIOUX          -0.762045   0.490535  -1.553 0.120330    
countyLUZERNE              -0.487915   0.226574  -2.153 0.031303 *  
countyLYCOMING              0.022862   0.279720   0.082 0.934861    
countyLYON                  0.499597   0.407980   1.225 0.220764    
countyMACON                -1.368647   0.490453  -2.791 0.005269 ** 
countyMADISON              -0.520410   0.268368  -1.939 0.052504 .  
countyMAHNOMEN             -0.028594   1.015322  -0.028 0.977533    
countyMARICOPA             -1.129202   0.210504  -5.364 8.27e-08 ***
countyMARIES               -2.699613   0.732752  -3.684 0.000230 ***
countyMARION               -0.302916   0.225803  -1.342 0.179780    
countyMARSHALL             -0.406955   0.354018  -1.150 0.250360    
countyMARTIN               -0.526669   0.353987  -1.488 0.136824    
countyMCDONALD             -2.246716   0.490451  -4.581 4.67e-06 ***
countyMCHENRY              -0.304204   0.275493  -1.104 0.269520    
countyMCINTOSH             -0.277287   0.390796  -0.710 0.478000    
countyMCKEAN               -0.954304   0.329879  -2.893 0.003824 ** 
countyMCKENZIE             -0.244516   0.455651  -0.537 0.591533    
countyMCLEAN                0.135445   0.317916   0.426 0.670087    
countyMCLEOD               -0.326694   0.344893  -0.947 0.343538    
countyMEEKER               -0.208094   0.490469  -0.424 0.671372    
countyMENOMINEE             0.063287   0.219903   0.288 0.773509    
countyMERCER               -0.120359   0.230523  -0.522 0.601600    
countyMIAMI                 0.005350   0.279726   0.019 0.984742    
countyMIDDLESEX            -0.498561   0.213272  -2.338 0.019420 *  
countyMIFFLIN              -0.343434   0.376507  -0.912 0.361703    
countyMILLE LACS           -0.817580   0.732742  -1.116 0.264538    
countyMILLER               -1.426228   0.376493  -3.788 0.000152 ***
countyMILLIE LACS          -1.261750   0.344908  -3.658 0.000255 ***
countyMISSISSIPPI          -2.440247   0.610142  -3.999 6.39e-05 ***
countyMOHAVE               -1.736913   0.230175  -7.546 4.80e-14 ***
countyMONITEAU             -1.813527   0.538454  -3.368 0.000759 ***
countyMONROE               -0.222523   0.231714  -0.960 0.336904    
countyMONTGOMERY           -0.378298   0.235501  -1.606 0.108222    
countyMONTOUR               0.442817   0.455649   0.972 0.331151    
countyMORGAN               -1.117911   0.329880  -3.389 0.000704 ***
countyMORRISON             -0.310771   0.390804  -0.795 0.426507    
countyMORTON                0.082952   0.230099   0.361 0.718476    
countyMOUNTRAIL             0.258335   0.303904   0.850 0.395312    
countyMOWER                 0.203813   0.344895   0.591 0.554569    
countyMURRAY                1.103634   1.015322   1.087 0.277067    
countyNAVAJO               -1.138828   0.245628  -4.636 3.58e-06 ***
countyNELSON                0.139625   0.284527   0.491 0.623630    
countyNEW MADRID           -1.543013   0.429068  -3.596 0.000324 ***
countyNEWTON               -0.664282   0.293101  -2.266 0.023445 *  
countyNICOLLET              0.775468   0.538463   1.440 0.149850    
countyNOBLE                -0.339941   0.303917  -1.119 0.263362    
countyNOBLES                0.538120   0.610138   0.882 0.377813    
countyNODAWAY              -0.265541   0.407983  -0.651 0.515148    
countyNORFOLK              -0.698656   0.221500  -3.154 0.001613 ** 
countyNORMAN               -0.346726   0.610130  -0.568 0.569854    
countyNORTHAMPTON           0.729678   0.284562   2.564 0.010353 *  
countyNORTHUMBERLAND        0.363910   0.330089   1.102 0.270284    
countyOHIO                 -0.631026   0.538458  -1.172 0.241254    
countyOLIVER                0.384836   0.308148   1.249 0.211737    
countyOLMSTED              -0.099364   0.293190  -0.339 0.734685    
countyONEIDA               -0.951849   0.235176  -4.047 5.21e-05 ***
countyORANGE               -0.393420   0.364374  -1.080 0.280291    
countyOREGON               -0.093011   0.610132  -0.152 0.878839    
countyOSAGE                -1.499105   0.455638  -3.290 0.001004 ** 
countyOTTER TAIL           -0.005530   0.407974  -0.014 0.989186    
countyOWEN                 -1.602658   0.490451  -3.268 0.001087 ** 
countyOZARK                -0.727211   0.490452  -1.483 0.138170    
countyPARKE                -0.771695   0.429054  -1.799 0.072107 .  
countyPEMBINA               0.499995   0.244337   2.046 0.040745 *  
countyPEMISCOT             -0.299655   1.015324  -0.295 0.767897    
countyPENNINGTON           -0.704389   0.610132  -1.154 0.248323    
countyPERRY                -0.488839   0.344895  -1.417 0.156404    
countyPETTIS               -0.867354   0.312791  -2.773 0.005563 ** 
countyPHELPS               -1.576189   0.336925  -4.678 2.93e-06 ***
countyPHILADELPHIA         -1.040991   0.225534  -4.616 3.96e-06 ***
countyPIERCE               -0.169446   0.317926  -0.533 0.594061    
countyPIKE                 -0.788542   0.287178  -2.746 0.006045 ** 
countyPIMA                 -1.345018   0.216384  -6.216 5.27e-10 ***
countyPINAL                -1.192385   0.270360  -4.410 1.04e-05 ***
countyPINE                 -0.731366   0.455642  -1.605 0.108490    
countyPIPESTONE             0.398219   0.538532   0.739 0.459647    
countyPLATTE               -0.094518   0.287183  -0.329 0.742071    
countyPLYMOUTH             -1.070867   0.223524  -4.791 1.68e-06 ***
countyPOLK                 -0.836962   0.308144  -2.716 0.006614 ** 
countyPOPE                 -0.110183   0.732749  -0.150 0.880476    
countyPORTER               -0.829522   0.233910  -3.546 0.000392 ***
countyPOSEY                -0.535115   0.455639  -1.174 0.240246    
countyPOTTER                0.278560   0.353961   0.787 0.431308    
countyPRAIRIE ISLAND       -0.255881   0.490445  -0.522 0.601867    
countyPULASKI              -1.582897   0.317971  -4.978 6.51e-07 ***
countyPUTNAM               -1.466333   0.390803  -3.752 0.000176 ***
countyRALLS                -1.398268   1.015324  -1.377 0.168487    
countyRAMSEY               -0.078078   0.250440  -0.312 0.755226    
countyRANDOLPH             -0.799986   0.329874  -2.425 0.015317 *  
countyRANSOM                0.422048   0.429050   0.984 0.325292    
countyRAY                  -0.702784   0.317923  -2.211 0.027085 *  
countyRED CLIFF            -1.762588   0.455653  -3.868 0.000110 ***
countyRED LAKE             -0.774998   0.241657  -3.207 0.001345 ** 
countyREDWOOD               0.468666   0.490448   0.956 0.339300    
countyRENVILLE              0.148403   0.353958   0.419 0.675028    
countyREYNOLDS             -0.396319   1.015322  -0.390 0.696293    
countyRICE                  0.401549   0.364377   1.102 0.270476    
countyRICHLAND              0.134789   0.253838   0.531 0.595424    
countyRIPLEY               -0.880107   0.317909  -2.768 0.005641 ** 
countyROCK                 -0.090453   0.732749  -0.123 0.901758    
countyROLETTE              -0.056172   0.303908  -0.185 0.853364    
countyROSEAU               -0.080330   0.336926  -0.238 0.811560    
countyRUSH                 -1.803733   1.015324  -1.777 0.075674 .  
countySAGINAW CHIPPEWA     -1.178781   0.265344  -4.442 8.97e-06 ***
countySALINE               -0.139429   0.390797  -0.357 0.721262    
countySANTA CRUZ           -0.942364   0.344969  -2.732 0.006309 ** 
countySARGENT               0.210727   0.376498   0.560 0.575693    
countySAULT ST. MARIE      -2.217608   0.246179  -9.008  < 2e-16 ***
countySCHUYLER             -0.556662   1.015322  -0.548 0.583522    
countySCHUYLKILL            0.243291   0.271716   0.895 0.370598    
countySCOTLAND             -1.652904   0.538465  -3.070 0.002148 ** 
countySCOTT                -0.960119   0.252940  -3.796 0.000148 ***
countySHAKOPEE-MDEWAKANTO  -0.643123   0.538454  -1.194 0.232349    
countySHANNON              -0.368983   0.455640  -0.810 0.418064    
countySHELBY               -0.175713   0.376567  -0.467 0.640782    
countySHERBURNE            -0.299549   0.407986  -0.734 0.462833    
countySHERIDAN              0.201907   0.490448   0.412 0.680582    
countySIBLEY               -0.147118   0.538463  -0.273 0.784690    
countySIOUX                -0.437033   0.732785  -0.596 0.550919    
countySLOPE                 0.099478   0.429052   0.232 0.816655    
countySNYDER                0.451619   0.455649   0.991 0.321629    
countySOKAOGAN CHIPPEWA    -0.610601   0.538480  -1.134 0.256844    
countySOMERSET             -0.650907   0.296428  -2.196 0.028122 *  
countySPENCER              -0.882444   0.364368  -2.422 0.015456 *  
countyST CHARLES           -0.849048   0.250443  -3.390 0.000701 ***
countyST CLAIR             -0.417023   0.610132  -0.683 0.494306    
countyST FRANCOIS          -0.887646   0.252064  -3.522 0.000431 ***
countyST JOSEPH            -0.444090   0.227198  -1.955 0.050648 .  
countyST LOUIS             -0.717824   0.218472  -3.286 0.001020 ** 
countyST LOUIS CITY        -0.870979   0.218472  -3.987 6.74e-05 ***
countyST. CROIX            -0.933138   0.344904  -2.705 0.006830 ** 
countySTARK                 0.261600   0.225963   1.158 0.247004    
countySTARKE               -1.254362   0.407976  -3.075 0.002112 ** 
countySTE GENEVIEVE        -1.521231   0.364368  -4.175 3.00e-05 ***
countySTEARNS               0.002452   0.287183   0.009 0.993187    
countySTEELE                0.088999   0.317921   0.280 0.779528    
countySTEUBEN               0.170333   0.344910   0.494 0.621423    
countySTEVENS               0.402188   0.732749   0.549 0.583100    
countySTOCKBRIDGE-MUNSEE    0.296578   0.252937   1.173 0.241003    
countySTODDARD             -1.335445   0.390851  -3.417 0.000636 ***
countySTONE                -0.940992   0.390797  -2.408 0.016060 *  
countySTUTSMAN              0.045857   0.260108   0.176 0.860062    
countySUFFOLK              -1.021282   0.236966  -4.310 1.65e-05 ***
countySULLIVAN             -1.356866   0.329880  -4.113 3.93e-05 ***
countySUSQUEHANNA          -0.330537   0.303898  -1.088 0.276767    
countySWIFT                -0.402531   0.538463  -0.748 0.454743    
countySWITZERLAND          -0.686937   0.732752  -0.937 0.348533    
countyTANEY                -0.873530   0.376554  -2.320 0.020368 *  
countyTEXAS                -1.434406   0.407976  -3.516 0.000440 ***
countyTIOGA                -0.425572   0.277534  -1.533 0.125202    
countyTIPPECANOE           -0.029220   0.261313  -0.112 0.910969    
countyTIPTON               -0.951878   0.490451  -1.941 0.052303 .  
countyTODD                  0.126190   0.610130   0.207 0.836150    
countyTOWNER                0.258824   0.376502   0.687 0.491815    
countyTRAILL                0.122467   0.284525   0.430 0.666893    
countyTRAVERSE              0.462899   0.538455   0.860 0.389982    
countyUNION                 0.767014   0.407980   1.880 0.060128 .  
countyUPPER SIOUX           0.255736   0.376494   0.679 0.496989    
countyVANDERBURGH          -0.919414   0.271708  -3.384 0.000717 ***
countyVENANGO              -0.187665   0.284580  -0.659 0.509621    
countyVERMILLION           -0.396170   0.407979  -0.971 0.331540    
countyVERNON               -1.410995   0.344895  -4.091 4.32e-05 ***
countyVIGO                 -0.227942   0.268348  -0.849 0.395661    
countyWABASH               -0.027189   0.329870  -0.082 0.934311    
countyWABASHA               0.319318   0.429062   0.744 0.456755    
countyWADENA               -0.353433   0.490445  -0.721 0.471147    
countyWALSH                 0.590069   0.251241   2.349 0.018859 *  
countyWARD                 -0.183495   0.240678  -0.762 0.445830    
countyWARREN               -0.707522   0.287180  -2.464 0.013765 *  
countyWARRICK              -1.384988   0.300009  -4.616 3.94e-06 ***
countyWASECA               -0.930657   0.538455  -1.728 0.083944 .  
countyWASHINGTON           -0.411924   0.228839  -1.800 0.071875 .  
countyWATONWAN              0.904046   0.610132   1.482 0.138440    
countyWAYNE                -0.564261   0.245537  -2.298 0.021575 *  
countyWEBSTER              -1.320250   0.329871  -4.002 6.31e-05 ***
countyWELLS                -0.709963   0.312790  -2.270 0.023238 *  
countyWESTMORELAND         -0.380048   0.239321  -1.588 0.112306    
countyWHITE                -1.171433   0.323607  -3.620 0.000296 ***
countyWHITE EARTH          -0.038566   0.287199  -0.134 0.893181    
countyWHITLEY              -0.255376   0.293098  -0.871 0.383609    
countyWILKIN                0.840443   1.015322   0.828 0.407822    
countyWILLIAMS             -0.011229   0.293109  -0.038 0.969442    
countyWINONA                0.102124   0.344894   0.296 0.767157    
countyWISCONSIN WINNEBAGO  -1.203646   0.303890  -3.961 7.51e-05 ***
countyWORCESTER            -0.306700   0.217877  -1.408 0.159253    
countyWORTH                -1.442251   0.732749  -1.968 0.049059 *  
countyWRIGHT               -0.369518   0.303893  -1.216 0.224028    
countyWYOMING              -0.553656   0.353963  -1.564 0.117804    
countyYAVAPAI              -1.486729   0.249728  -5.953 2.70e-09 ***
countyYELLOW MEDICINE      -0.203049   0.732749  -0.277 0.781703    
countyYORK                  0.638449   0.240683   2.653 0.007996 ** 
countyYUMA                 -1.824017   0.268522  -6.793 1.15e-11 ***
floor                      -0.096664   0.007033 -13.745  < 2e-16 ***
typebldg                   -0.165668   0.017538  -9.446  < 2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 0.9939 on 12389 degrees of freedom
Multiple R-squared:  0.2907,	Adjusted R-squared:  0.2685 
F-statistic: 13.12 on 387 and 12389 DF,  p-value: < 2.2e-16

> z <- lm(radon ~ floor + typebldg, data = radon)
> summary(z)

Call:
lm(formula = radon ~ floor + typebldg, data = radon)

Residuals:
    Min      1Q  Median      3Q     Max 
-4.1502 -0.7300 -0.0024  0.7300  4.7933 

Coefficients:
             Estimate Std. Error t value Pr(>|t|)    
(Intercept)  1.154429   0.023758   48.59   <2e-16 ***
floor       -0.188644   0.007512  -25.11   <2e-16 ***
typebldg    -0.235770   0.019562  -12.05   <2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 1.133 on 12774 degrees of freedom
Multiple R-squared:  0.04999,	Adjusted R-squared:  0.04985 
F-statistic: 336.1 on 2 and 12774 DF,  p-value: < 2.2e-16

> library(lme4)
Loading required package: Matrix
> zc <- lmer(radon ~ floor + typebldg + (1|county), data = radon)
> summary(zc)
Linear mixed model fit by REML ['lmerMod']
Formula: radon ~ floor + typebldg + (1 | county)
   Data: radon

REML criterion at convergence: 36855.6

Scaled residuals: 
    Min      1Q  Median      3Q     Max 
-4.8701 -0.5880  0.0128  0.6086  4.9877 

Random effects:
 Groups   Name        Variance Std.Dev.
 county   (Intercept) 0.3396   0.5828  
 Residual             0.9881   0.9940  
Number of obs: 12777, groups:  county, 386

Fixed effects:
             Estimate Std. Error t value
(Intercept)  1.078113   0.038677  27.875
floor       -0.102853   0.007005 -14.683
typebldg    -0.169967   0.017485  -9.721

Correlation of Fixed Effects:
         (Intr) floor 
floor    -0.203       
typebldg -0.485  0.255
> zcw <- lmer(radon ~ floor + typebldg + (1|county) + (1|wave), data = radon)
> summary(zcw)
Linear mixed model fit by REML ['lmerMod']
Formula: radon ~ floor + typebldg + (1 | county) + (1 | wave)
   Data: radon

REML criterion at convergence: 36829.9

Scaled residuals: 
    Min      1Q  Median      3Q     Max 
-4.7942 -0.5904  0.0113  0.6028  5.0171 

Random effects:
 Groups   Name        Variance Std.Dev.
 county   (Intercept) 0.325611 0.57062 
 wave     (Intercept) 0.009425 0.09708 
 Residual             0.980985 0.99045 
Number of obs: 12777, groups:  county, 386; wave, 135

Fixed effects:
             Estimate Std. Error t value
(Intercept)  1.072987   0.039626  27.078
floor       -0.103019   0.007011 -14.694
typebldg    -0.170289   0.017471  -9.747

Correlation of Fixed Effects:
         (Intr) floor 
floor    -0.197       
typebldg -0.472  0.254

> zcwSlope <- lmer(radon ~ floor + typebldg + (1|county) + (floor|wave), data = radon)
> summary(zcwSlope)
Linear mixed model fit by REML ['lmerMod']
Formula: radon ~ floor + typebldg + (1 | county) + (floor | wave)
   Data: radon

REML criterion at convergence: 36766.8

Scaled residuals: 
    Min      1Q  Median      3Q     Max 
-4.8636 -0.5927  0.0105  0.6056  5.0344 

Random effects:
 Groups   Name        Variance Std.Dev. Corr
 county   (Intercept) 0.321956 0.56741      
 wave     (Intercept) 0.005531 0.07437      
          floor       0.010106 0.10053  0.25
 Residual             0.968386 0.98407      
Number of obs: 12777, groups:  county, 386; wave, 135

Fixed effects:
            Estimate Std. Error t value
(Intercept)  1.07908    0.03887   27.76
floor       -0.13066    0.01245  -10.49
typebldg    -0.16346    0.01743   -9.38

Correlation of Fixed Effects:
         (Intr) floor 
floor    -0.087       
typebldg -0.478  0.134

> anova(zcwSlope, zcw)
refitting model(s) with ML (instead of REML)
Data: radon
Models:
zcw: radon ~ floor + typebldg + (1 | county) + (1 | wave)
zcwSlope: radon ~ floor + typebldg + (1 | county) + (floor | wave)
         Df   AIC   BIC logLik deviance  Chisq Chi Df Pr(>Chisq)    
zcw       6 36823 36867 -18405    36811                             
zcwSlope  8 36765 36824 -18374    36749 61.983      2  3.472e-14 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

> zcw <- lmer(radon ~ floor + typebldg + (1|county) + (wave|county), data = radon)
Error: number of observations (=12777) <= number of random effects (=52110) for term (wave | county); the random-effects parameters and the residual variance (or scale parameter) are probably unidentifiable

# Produce new responses for the fitted model zcw
> zcwYSim <- simulate(zcw, nsim = 100)
# Fit the same model to the new responses. 
# Put each new model in a list.
> nsim=100
> zcwSim <- vector('list', length = 100)
>  for (i in 1:nsim) {
+  zcwSim[[i]] <- refit(zcw, newresp =
+  zcwYSim[,i])
+  }

# Exam data analysis
> library(mlmRev)
> names(Exam)
 [1] "school"   "normexam" "schgend"  "schavg"   "vr"       "intake"  
 [7] "standLRT" "sex"      "type"     "student" 
# Null model
> lmer(normexam ~ 1 + (1 | school), data=Exam)
Linear mixed model fit by REML ['lmerMod']
Formula: normexam ~ 1 + (1 | school)
   Data: Exam
REML criterion at convergence: 11014.65
Random effects:
 Groups   Name        Std.Dev.
 school   (Intercept) 0.4142  
 Residual             0.9207  
Number of obs: 4059, groups:  school, 65
Fixed Effects:
(Intercept)  
   -0.01325  

> lmer(normexam ~ standLRT + (1 | school), data=Exam)
Linear mixed model fit by REML ['lmerMod']
Formula: normexam ~ standLRT + (1 | school)
   Data: Exam
REML criterion at convergence: 9368.765
Random effects:
 Groups   Name        Std.Dev.
 school   (Intercept) 0.3063  
 Residual             0.7522  
Number of obs: 4059, groups:  school, 65
Fixed Effects:
(Intercept)     standLRT  
   0.002323     0.563307  
   
 > lmer(normexam ~ standLRT + (standLRT | school), data=Exam, method="ML")
Linear mixed model fit by REML ['lmerMod']
Formula: normexam ~ standLRT + (standLRT | school)
   Data: Exam
REML criterion at convergence: 9327.6
Random effects:
 Groups   Name        Std.Dev. Corr
 school   (Intercept) 0.3035       
          standLRT    0.1223   0.49
 Residual             0.7441       
Number of obs: 4059, groups:  school, 65
Fixed Effects:
(Intercept)     standLRT  
   -0.01165      0.55653  
Warning message:
Argument ‘method’ is deprecated. Use the REML argument to specify ML or REML estimation. 

> lmer(normexam ~ standLRT + schavg + (1 + standLRT | school), data=Exam)
Linear mixed model fit by REML ['lmerMod']
Formula: normexam ~ standLRT + schavg + (1 + standLRT | school)
   Data: Exam
REML criterion at convergence: 9323.885
Random effects:
 Groups   Name        Std.Dev. Corr
 school   (Intercept) 0.2778       
          standLRT    0.1238   0.37
 Residual             0.7440       
Number of obs: 4059, groups:  school, 65
Fixed Effects:
(Intercept)     standLRT       schavg  
  -0.001423     0.552242     0.294731  

> lmer(normexam ~ standLRT * schavg + (1 + standLRT | school), data=Exam)
Linear mixed model fit by REML ['lmerMod']
Formula: normexam ~ standLRT * schavg + (1 + standLRT | school)
   Data: Exam
REML criterion at convergence: 9320.387
Random effects:
 Groups   Name        Std.Dev. Corr
 school   (Intercept) 0.2763       
          standLRT    0.1106   0.36
 Residual             0.7442       
Number of obs: 4059, groups:  school, 65
Fixed Effects:
    (Intercept)         standLRT           schavg  standLRT:schavg  
      -0.007092         0.557943         0.373398         0.161829  
