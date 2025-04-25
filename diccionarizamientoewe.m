function diccionarios = ordenar(nombre_archivo,n)

    lista_dict = {};
    
    % Cargar datos
    load(nombre_archivo);
    
    % Asegurar que la señal es un vector fila
    senal = reshape(senal, 1, []);
    
    % Recorrer la señal en ventanas deslizantes
    for i = 1:(length(senal) - n + 1)
        ventana = senal(i:i+n-1);
        feat = caracteristicas(ventana);
        lista_dict{end+1}= feat;
        % También puedes calcular la entropía, FFT, etc.
        % pausa opcional si es en tiempo real
    end

    diccionarios = lista_dict;
end    