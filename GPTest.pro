
PRO GPTest, N, kernel_type, output_type, par1, par2, par3, par4, par5, DEBUG=debug
    ; Default number of points
    IF (SIZE(N, /TNAME) EQ 'UNDEFINED') THEN N = 100L ELSE N = LONG(N)

    ; Generate x, y, and diag_
    x = FINDGEN(N) * 0.1D
    y = SIN(x)
    diag_ = REPLICATE(0.1D, N)  ; yerr equivalent
	print,x,y,diag_
    ; Initialize result array
    result = DBLARR(N)

    ; Define kernel parameters
    IF (SIZE(par1, /TNAME) EQ 'UNDEFINED') THEN par1 = 1.0D ELSE par1 = DOUBLE(par1)
    IF (SIZE(par2, /TNAME) EQ 'UNDEFINED') THEN par2 = 0.5D ELSE par2 = DOUBLE(par2)
    IF (SIZE(par3, /TNAME) EQ 'UNDEFINED') THEN par3 = 0.01D ELSE par3 = DOUBLE(par3)
    IF (SIZE(par4, /TNAME) EQ 'UNDEFINED') THEN par4 = 1.0D-5 ELSE par4 = DOUBLE(par4)
    IF (SIZE(par5, /TNAME) EQ 'UNDEFINED') THEN par5 = 0.0D ELSE par5 = DOUBLE(par5)

    ; Loop over kernel_type and output_type
    FOR kernel_type = 0, 4 DO BEGIN
        FOR output_type = 0, 1 DO BEGIN
            if kernel_type eq 0 then kernel_name = 'RealTerm'
            if kernel_type eq 1 then kernel_name = 'ComplexTerm'
            if kernel_type eq 2 then kernel_name = 'SHOTerm'
            if kernel_type eq 3 then kernel_name = 'Matern32Term'
            if kernel_type eq 4 then kernel_name = 'RotationTerm'
            
            if output_type eq 0 then output_name = 'log-likelihood'
            if output_type eq 1 then output_name = 'prediction'
  
     
            result = computeGP(N, x, y, diag_, kernel_type, output_type, par1, par2, par3, par4, par5)
            if output_type eq 0 then print, kernel_name, ' ', output_name, ' = ', result[0]
            if output_type eq 1 then print, kernel_name, ' ', output_name,  '= ', result[0:4]
        ENDFOR
    ENDFOR

END

