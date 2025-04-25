function salida = update_plot(new_data)
    persistent hFigure hPlot dataBuffer
    
    % Tamaño máximo de la tabla
    maxDataPoints = 50;  % Cambia esto según cuántos puntos quieras en la tabla

    % Inicialización de la gráfica y buffer de datos si es la primera vez
    if isempty(hFigure) || ~isvalid(hFigure)
        % Crear la figura y el gráfico inicial
        hFigure = figure('Name', 'Gráfica Dinámica');
        hPlot = plot(1, new_data(1), 'o-', 'MarkerSize', 6, 'LineWidth', 2);
        title('Datos Dinámicos');
        xlabel('Índice');
        ylabel('Valor');
        dataBuffer = new_data;  % Buffer de datos inicial con el primer punto
        salida =""
        % Guardar los datos en el workspace base
        assignin('base', 'receivedData', new_data);  % Guardar en el workspace base
    else
        % Mover la tabla hacia la derecha (desplazar los datos hacia la izquierda)
        dataBuffer = [dataBuffer, new_data(end)];  % Añadir el nuevo dato al final del buffer
        % Si el tamaño del buffer excede el número máximo de puntos, eliminar el más antiguo
        if length(dataBuffer) > maxDataPoints
            dataBuffer = dataBuffer(2:end);  % Eliminar el primer valor (más antiguo)
        end
        
        % Actualizar los datos de la gráfica
        set(hPlot, 'YData', dataBuffer);
        set(hPlot, 'XData', 1:length(dataBuffer));  % Ajustar los valores del eje X según el número de puntos
        
        % Obtener los datos actuales de la variable 'receivedData' en el workspace base
        receivedData = evalin('base', 'receivedData');  % Obtener los datos existentes en el workspace base
        
        % Añadir el nuevo dato a los datos existentes
        receivedData = [receivedData, new_data];  % Añadir los nuevos datos a la variable 'receivedData'
        salida = caracteristicas(new_data);
        % Actualizar la figura
        drawnow;
    end

end

