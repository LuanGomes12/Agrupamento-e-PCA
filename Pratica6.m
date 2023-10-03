% ********** Luan Gomes Magalhães Lima - 473008                      **********
% ********** Tópicos Especiais em Telecomunicações 1 - Prática 6     **********

% Inicializações
clear all;
close all;
clc;

%% Análise Exploratória
% Leitura da Base de Dados
base = readtable("Customers.csv");

% Numerizar os atributos Gender e Profession
base.Gender = grp2idx(base.Gender);
base.Profession = grp2idx(base.Profession);

% Eliminar as amostras que possuem um dos valores de atributos inválidos (Nan)
ind_invalido = any(isnan(base.Variables), 2);
base = base(~ind_invalido, :);
base = table2array(base); % Converter a base de tabela para matriz

%% Agrupamento - Utilizando apenas K-means, sem PCA
% Inicializações dos critérios de avaliação do agrupamento
melhor_SWC = 0;
melhor_k = 0;

% Matriz para armazenar os melhores SWC e K
matriz_final = zeros(10, 2);

% Laço para rodar o algoritmo 10 vezes e pegar o melhor resultado em termos
% de Largura Média de Silhueta (SWC)
for z = 1 : 10
    % Laço para percorrer os valores de k
    for k = 2 : 5
        % Valor de k atual
        k_atual = k;

        % Escolhas das sementes aleatoriamente
        ind_sementes = randperm(size(base, 1), k_atual);
        sementes = base(ind_sementes, :);
        centroides_atual = sementes;

        % Critério de convergência
        convergencia = false;

        % Definir número de iterações -> (CASO ALTERNATIVO)
        iteracoes = 0;

        % Laço para verificar o critério de convergência
        while ~convergencia
            % Atribuir cada amostra ao cluster de centro mais próximo
            % Utiliza-se a Distância Euclidiana como medida de similaridade
            distancias = pdist2(base, centroides_atual, 'euclidean');
            [~, cluster_atual] = min(distancias, [], 2);

            % Recalcular os centroides
            centroides_antigo = centroides_atual;
            for i = 1 : k
                centroides_atual(i, :) = mean(base(cluster_atual == i, :), 1);
            end

            % Verificar critério de convergência
            if isequal(centroides_antigo, centroides_atual)
                convergencia = true;
            end

            % Caso os centroides não se estabilizem, utiliza-se o critério do
            % número máximo de iterações (num_max = 100) -> (CASO ALTERNATIVO)
            iteracoes = iteracoes + 1;
            if iteracoes >= 100
                break;
            end

        end

        % Cálculo da Largura Média de Silhueta (SWC)
        SWC = silhouette(base, cluster_atual);
        SWC_media_atual = mean(SWC);

        % Verificar o melhor SWC e o melhor K
        if SWC_media_atual > melhor_SWC
            melhor_SWC = SWC_media_atual;
            melhor_k = k_atual;
        end
    end

    % Inserir valores na matriz final que será analisada para pegar o maior
    % SWC
    matriz_final(z, 1) = melhor_SWC;
    matriz_final(z, 2) = melhor_k;

end

% Pegar o melhor SWC e K após o algoritmo rodar 10 vezes
ind_melhor = max(matriz_final(1, :));
melhor_SWC_final = matriz_final(ind_melhor, 1);
melhor_k_final = matriz_final(ind_melhor, 2);

% Saída para o agrupamento utilizando apenas K-means
fprintf('Agrupamento de dados usando apenas K-means\n');
fprintf('Resultado do melhor SWC: %.6f\n', melhor_SWC_final);
fprintf('Resultado do melhor K: %d\n', melhor_k_final);
fprintf('--------------------------------------------\n');

%% Agrupamento - Utilizando PCA e K-means
% Matriz para armazenar os melhores SWC, K e número de componentes para PCA
matriz_final = zeros(10, 3);

% Laço para rodar o algoritmo 10 vezes e pegar o melhor resultado em termos
% de Largura Média de Silhueta (SWC)
for z = 1 : 10
    % Implementação do PCA
    base_pca = base;

    % Cálculo da Matriz de Covariâncias
    matriz_cov = cov(base_pca);

    % Cálculo dos autovetores e autovalores
    [autovetores, autovalores] = eig(matriz_cov);
    autovalores = diag(autovalores);

    % Ordenar os autovetores em ordem descrescente dos autovalores
    [autovalores_ord, ind_ord] = sort(autovalores, 'descend');
    autovetores_ord = autovetores(:, ind_ord);

    % Laço para percorrer diversos valores para o número de componentes do PCA
    % Possíveis valores = [1, 2, 3, 4, 5, 6, 7, 8]
    num_componentes_max = length(base(1, :));

    % Inicializações dos critérios de avaliação do agrupamento
    melhor_SWC = 0;
    melhor_k = 0;
    melhor_qtd_componentes_pca = 0;
    
    % Laço para percorrer todos os possíveis valores do número de
    % componentes do PCA
    for num_componentes = 1 : num_componentes_max
        % Número de componentes atual do PCA
        componentes = autovetores_ord(:, 1 : num_componentes);

        % Projeção dos dados
        dados_projetados = base_pca * componentes;

        % Laço para percorrer os valores de k
        for k = 2  : 5
            % Valor de k atual
            k_atual = k;

            % Escolhas das sementes aleatoriamente
            ind_sementes = randperm(size(dados_projetados, 1), k_atual);
            sementes = dados_projetados(ind_sementes, :);
            centroides_atual = sementes;

            % Critério de convergência
            convergencia = false;

            % Definir número de iterações -> (CASO ALTERNATIVO)
            iteracoes = 0;

            % Laço para verificar o critério de convergência
            while ~convergencia
                % Atribuir cada amostra ao cluster de centro mais próximo
                % Utiliza-se a Distancia Euclidiana como medida de similaridade
                distancias = pdist2(dados_projetados, centroides_atual, 'euclidean');
                [~, cluster_atual] = min(distancias, [], 2);

                % Recalcular os centroides
                centroides_antigo = centroides_atual;
                for i = 1 : k
                    centroides_atual(i, :) = mean(dados_projetados(cluster_atual == i), 1);
                end

                % Verificar critério de convergência
                if isequal(centroides_antigo, centroides_atual)
                    convergencia = true;
                end

                % Caso os centroides não se estabilizem, utiliza-se o critério do
                % número máximo de iterações (num_max = 100) -> (CASO ALTERNATIVO)
                iteracoes = iteracoes + 1;
                if iteracoes >= 100
                    break;
                end

            end

            % Cálculo da Largura Média de Silhueta (SWC)
            SWC = silhouette(dados_projetados, cluster_atual);
            SWC_media_atual = mean(SWC);

            % Verificar o melhor SWC, K e número de componentes para o PCA
            if SWC_media_atual > melhor_SWC
                melhor_SWC = SWC_media_atual;
                melhor_k = k_atual;
                melhor_qtd_componentes_pca = num_componentes;
            end
        end
    end

    % Inserir valores na matriz final que será analisada para pegar o maior
    % SWC
    matriz_final(z, 1) = melhor_SWC;
    matriz_final(z, 2) = melhor_k;
    matriz_final(z, 3) = melhor_qtd_componentes_pca;

end

% Pegar o melhor SWC, K e número de componentes para PCA após o algoritmo rodar 10 vezes
ind_melhor = max(matriz_final(1, :));
melhor_SWC_final = matriz_final(ind_melhor, 1);
melhor_k_final = matriz_final(ind_melhor, 2);
melhor_qtd_componentes_pca_final = matriz_final(ind_melhor, 3);

% Saída para o agrupamento utilizando PCA e K-means
fprintf('Agrupamento de dados usando PCA e K-means\n');
fprintf('Resultado do melhor SWC: %.6f\n', melhor_SWC_final);
fprintf('Resultado do melhor K: %d\n', melhor_k_final);
fprintf('Resultado do melhor número de componentes: %d\n', melhor_qtd_componentes_pca_final);

%% RESULTADOS FINAIS E OBSERVAÇÕES
% Com base nas análises e observações feitas, verifica-se que ao
% normalizar a base de dados, seja para o K-means sem PCA ou com PCA, o
% índice de Largura Média de Silhueta (SWC) diminui, o que afeta a qualidade
% do agrupamento. No caso do K-means sem PCA, a redução do SWC é bem maior,
% indicando uma qualidade ruim do agrupamento. Logo, a fim de garantir uma
% melhor qualidade do agrupamento, eu não utilizei normalização dos dados
% em ambos os casos.