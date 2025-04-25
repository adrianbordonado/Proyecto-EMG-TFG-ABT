function salida = caracteristicas(datos)

    [~,n] = size(datos);
    fs = 500;
    F = fft(datos);

    %ENTROPÍA

    % Calcular histograma
    [counts, ~] = hist(datos, 20);
    
    % Normalizar para obtener distribución de probabilidad
    P = counts / sum(counts + eps);
    
    % Calcular entropía de Shannon , EN ESTE CASO EN EL DOMINIO DEL TIEMPO
    ent = -sum(P .* log2(P + eps));

    %FRECUENCIA MEDIA

    P2 = abs(F/n);
    P1 = P2(1:n/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    fmed = mean(P1);

    %FRECUENCIA PREDOMINANTE

    [psd_welch,w1] = pwelch(datos);
    fpred = 1;

    %AGUDEZA

    ag= kurtosis(datos);

    %ASIMETRÍA

    as = skewness(datos);

    %RMS 

    rms =1;

    %POTENCIA MEDIA

    pwmed= mean(psd_welch);

    %VALOR ABSOLUTO MEDIO

    mav = mean(abs(datos));;


    etiquetas = ["entropia","frecuencia media", "frecuencia predominante","agudeza","asimetria", "rms", "potencia media", "mav"];
    valores = [ent, fmed, fpred, ag, as, rms, pwmed, mav];
    salida = dictionary(etiquetas,valores);

end