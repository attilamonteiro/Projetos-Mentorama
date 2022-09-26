/*Tarefa SQL Módulo 06
A sintaxe CASE WHEN é uma das mais utilizadas em SQL no dia-a-dia, seja de
análise de dados ou gerenciamento de bancos de dados. Para fixar seu
aprendizado, utilize a base de dados sakila e responda:

1. A equipe comercial da locadora está criando uma nova estratégia e pediu que
você classificasse os filmes do catálogo de acordo com seu preço de aluguel.

A regra para a classificação é:
- Aluguel menor ou igual a 0.99: Básico
- Aluguel entre 0.99 e 2.99: Essencial
- Aluguel maior que 2.99: Exclusivo
Escreva uma consulta SQL que, a partir da tabela film, retorne a lista
classificada, contendo as colunas: id, nome do filme, preço de aluguel e
classificação.
  */
SELECT film_id as 'id', title as 'nome do filme', rental_rate as 'preço do aluguel', rating as 'classificação',
	(CASE 
    WHEN rental_rate<=0.99 THEN 'Básico'
    WHEN rental_rate between 0.99 and 2.99 THEN 'Essencial'
    ELSE 'Exclusivo' END) as 'Classificação de aluguel'
FROM sakila.film;
/*
2 – 
A tabela customer contém: dados das pessoas cadastradas na rede de
locadoras, identificação da loja que a pessoa se cadastrou (a partir da coluna
store_id) e também se o cadastro está ativo ou não (coluna active).
Escreva uma consulta SQL que utilize o comando CASE WHEN e retorne a
contagem de consumidores ativos e inativos para cada uma das lojas.
Dica: existem 4 possibilidades de resultados (lojas 1 e 2, e pessoa
ativa ou inativa)
*/
SELECT 
		COUNT( CASE 
        WHEN active = 1 AND store_id = 1 THEN "Ativos store 1"
        ELSE NULL END) AS ativos_store_1,
		COUNT( CASE 
        WHEN active = 0 AND store_id = 1 THEN "Inativos"
        ELSE NULL END) AS inativos_store_1,
		COUNT( CASE 
        WHEN active = 1 AND store_id = 2 THEN "Ativos store 2"
        ELSE NULL END) AS ativos_store_2,
		COUNT( CASE 
        WHEN active = 0 AND store_id = 2 THEN "Inativos"
        ELSE NULL END) AS inativos_store_2

FROM sakila.customer;


/*
3 – 

Utilizando os conceitos já estudados em módulos anteriores, qual outra
estratégia poderia ser utilizada para atingir o mesmo resultado da pergunta
acima? Escreva a consulta SQL.
*/
SELECT store_id as Lojas,
		sum(active = 0) as Inativa,
        sum(active = 1) as Ativa

FROM sakila.customer
group by store_id;

4 – 

Na tabela address, a coluna city_id representa a cidade na qual aquele
endereço está situado. Como você pode já ter percebido, ela faz referência à
tabela city, que também contém uma coluna de mesmo nome.
Consultando manualmente o nome da cidade para os 5 primeiros registros da
tabela address (registros com address_id entre 1 e 5), escreva uma consulta
SQL que retorne o endereço e o nome da cidade à qual ele se refere.
Dica: use a sintaxe CASE WHEN, acompanhada da relação que você
descobrir entre os valores de city_id nas duas tabelas.
*/
SELECT 
ad.address_id as'id de enrereço', ad.address as 'endereço', ci.city as'cidade do endereço'
FROM address ad
INNER JOIN city ci on ci.city_id=ad.city_id
WHERE address_id < 6;

/*
OU
*/
SELECT address_id, address AS Endereco, 
(CASE WHEN city_id = 300 THEN "Lethbridge" 
           WHEN city_id = 463 THEN "Sasebo"  
           WHEN city_id = 576 THEN "QLD" END ) AS nome_cidade 
FROM address 
WHERE address_id < 6;
