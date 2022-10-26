/* Tarefa SQL Módulo 09
Conhecendo subconsultas, seu arsenal de ferramentas em SQL fica ainda mais completo. 
Você agora pode juntar muitas camadas de processamento de dados em uma só consulta e reduzir o número de passos até respostas a perguntas mais complexas. 
Pratique os conhecimentos em subconsultas e responda às seguintes perguntas. Submeta suas respostas em um arquivo com extensão .sql, e use comentários (com os caracteres -- ou /* *,/) 
para escrever suas respostas por extenso e delimitar as questões. 

1. Se você é do tempo da Blockbuster, deve lembrar que os filmes mais famosos ou mais novos eram oferecidos em maior quantidade para aluguel -- já que, naturalmente, a demanda por eles era maior que a média. Vamos verificar essa premissa na nossa locadora usando o banco de dados sakila! 


a)	 Escreva uma consulta que liste, em ordem decrescente, os filmes mais alugados na nossa locadora. Utilize a tabela rental para isso, relacionando com a tabela inventory. 
*/

use sakila;
SELECT f.title AS 'nome do filme',
i.film_id AS 'id do filme',
count(i.film_id) AS 'quantidades disponiveis'
FROM rental r 
INNER JOIN inventory i on i.inventory_id=r.inventory_id 
INNER JOIN film f on f.film_id=i.film_id 
GROUP BY 1 
ORDER BY 3 desc;

/*
b)  Escreva uma consulta que liste, em ordem decrescente, o número de unidades disponíveis de cada filme na locadora. 
*/

SELECT 
f.title AS 'nome do filme',
i.film_id AS 'id do filme',
count(i.film_id) AS 'quantidades disponiveis'
FROM inventory i
INNER JOIN film f on f.film_id=i.film_id
GROUP by 1
ORDER by 3 desc;


/*
c) Usando subconsultas, relacione as tabelas resultantes dos itens a) e b) e responda: os títulos mais alugados de fato correspondem aos que têm maior número de itens disponíveis? Qual consulta você usou para chegar a esse resultado? Bônus: caso queira, use métricas de descrição estatística (média, percentil) para responder à pergunta de forma mais embasada!
*/


WITH aux_rental as 
(SELECT 
f.title AS film_name,
i.film_id AS id_film, 
count(r.rental_id) AS Qnt_alugueis
FROM rental r
INNER JOIN inventory i on i.inventory_id=r.inventory_id
INNER JOIN film f on f.film_id=i.film_id
GROUP BY 1
ORDER BY 3 desc)

#tabela auxiliar para quantidade de filmes disponiveis :

,aux_disponivel as
(SELECT 
f.title AS film_name,
i.film_id AS id_film,
count(i.film_id) AS Qnt_disponiveis
FROM inventory i
INNER JOIN film f on f.film_id=i.film_id
GROUP by 1
ORDER by 3 desc)

#junção das tabelas auxiliares:

SELECT 
aux.film_name as "nome do filme",
aux.id_film as "codigo do filme",
aux.Qnt_disponiveis as "quantidade disponível",
aux1.Qnt_alugueis as "Quantos foram alugados",
(aux1.Qnt_alugueis/( SELECT count(*) from rental ))*100 as perc #porcentagem da qtd de alugueis de cada filme em relação ao total de alugueis
FROM aux_disponivel aux
INNER JOIN aux_rental aux1 on aux.id_film=aux1.id_film
GROUP BY 1
ORDER BY 5 desc;

# OS TÍTULOS MAIS ALUGADOS FORAM OS QUE TINHAM MAIS ESTOQUE, PORÉM PODERIAM TER MAIS PARA ALUGAR

/*
2 - A view chamada sales_by_film_category traz o resultado total de vendas em aluguéis por categoria dos filmes. Escreva uma consulta que complemente essa tabela com as seguintes métricas para cada categoria: 
● Valor médio do aluguel dos filmes por categoria; ● Valor médio do custo de reposição dos filmes por categoria; ● Total de vendas por categoria na loja 1; ● Total de vendas por categoria na loja 2; (dica: use o comando CASE WHEN) ● Total de títulos disponíveis por categoria (dica: use a tabela inventory)
*/

WITH sales_by_film_category AS 
(SELECT 
ca.name AS category,
ca.category_id AS category_ID,
SUM(p.amount) AS total_sales
FROM payment AS p
INNER JOIN rental AS r ON p.rental_id = r.rental_id
INNER JOIN inventory AS i ON r.inventory_id = i.inventory_id
INNER JOIN film AS f ON i.film_id = f.film_id
INNER JOIN film_category AS fc ON f.film_id = fc.film_id
INNER JOIN category AS ca ON fc.category_id = ca.category_id
GROUP BY ca.name
ORDER BY total_sales DESC)

,aux_payment AS # select da media de alugueis por categoria 
(SELECT 
   ca.name AS category, 
   ca.category_id as category_ID,
   count(f.film_id) AS qtd_filmes,
   AVG(p.amount) AS média_aluguel
FROM category ca
  INNER JOIN film_category fc 
    on fc.category_id=ca.category_id
  INNER JOIN film f 
    on f.film_id=fc.category_id
  INNER JOIN inventory i 
    on i.film_id=f.film_id
  INNER JOIN rental r 
    on r.inventory_id=i.inventory_id
  INNER JOIN payment p 
    on p.rental_id=r.rental_id
GROUP by 2)
	
, aux_repos AS #select da media de reposição
(SELECT
ca.name AS category,
ca.category_id as category_ID,
count(f.film_id) AS qtd_filmes,
AVG(f.replacement_cost) AS média_reposição
FROM category ca
  INNER JOIN film_category fc
    on fc.category_id=ca.category_id
  INNER JOIN film f
    on f.film_id=fc.film_id
  INNER JOIN inventory i 
    on i.film_id=f.film_id
  INNER JOIN rental r 
    on r.inventory_id=i.inventory_id
  INNER JOIN payment p 
    on p.rental_id=r.rental_id
GROUP BY 2)

, aux_store AS # categoria por loja
(SELECT
ca.name AS category,
ca.category_id as category_ID,
count(f.film_id) AS qtd_filmes,
(CASE s.store_id WHEN 1 THEN "loja 1" WHEN 2 THEN "loja 2" END) AS loja
FROM category ca
  INNER JOIN film_category fc
    on fc.category_id=ca.category_id
  INNER JOIN film f
    on f.film_id=fc.film_id
  INNER JOIN inventory i 
    on i.film_id=f.film_id
  INNER JOIN store s
    on s.store_id=i.store_id
GROUP BY 4,1)

, aux_titulos AS # titulo por categoria
(SELECT
ca.name AS category,
ca.category_id as category_ID,
count(i.film_id) AS qtd_titulos
FROM category ca
  INNER JOIN film_category fc
    on fc.category_id=ca.category_id
  INNER JOIN film f
    on f.film_id=fc.film_id
  INNER JOIN inventory i 
    on i.film_id=f.film_id
GROUP BY 1)

#consulta final

SELECT 

aux.category, 
aux.total_sales,
aux1.média_aluguel,
aux2.média_reposição, 
aux3.loja, 
aux4.qtd_titulos


FROM sales_by_film_category aux
  INNER JOIN aux_payment aux1
    on aux.category_ID=aux1.category_ID
  INNER JOIN aux_repos aux2
    on aux1.category_ID=aux2.category_ID
  INNER JOIN aux_store aux3
    on aux2.category_ID=aux3.category_ID
  INNER JOIN aux_titulos aux4
    on aux3.category_ID=aux4.category_ID

;


/*
SOLUÇÃO 2 - Bonus:
*/

WITH loja AS
(
SELECT 
     ca.name AS categoria, 
     count(r.rental_id) AS vendas,
     (CASE WHEN store_id=1 THEN "LOJA 1" ELSE "LOJA 2" END) AS qual_loja
FROM category ca
 INNER JOIN film_category fc
    on fc.category_id=ca.category_id
     INNER JOIN film f
    on f.film_id=fc.film_id
     INNER JOIN inventory i 
    on i.film_id=f.film_id
     INNER JOIN rental r
    on r.inventory_id=i.inventory_id
GROUP BY 1,3)



,qtd_disponivel AS
(
SELECT ca.name AS categoria,
       COUNT(i.inventory_id) AS 'quant_disponivel'
  FROM inventory i
   INNER JOIN film f
       on f.film_id=i.film_id
       INNER JOIN film_category fc
       ON f.film_id = fc.film_id
       INNER JOIN category ca
       ON ca.category_id = fc.category_id
       LEFT JOIN (SELECT inventory_id,
                         return_date
                    FROM rental
                   GROUP BY 1) comp
       ON i.inventory_id = comp.inventory_id
 WHERE comp.return_date IS NOT NULL # só estão disponivéis os que foram entregues
 GROUP BY 1
 ORDER BY 2 DESC
 )
 
 ,sales as
 (SELECT
 ca.name AS categoria,
    SUM(p.amount) AS total_sales
  FROM payment AS p
    INNER JOIN rental AS r ON p.rental_id = r.rental_id
    INNER JOIN inventory AS i ON r.inventory_id = i.inventory_id
INNER JOIN film AS f ON i.film_id = f.film_id
    INNER JOIN film_category AS fc ON f.film_id = fc.film_id
    INNER JOIN category AS ca ON fc.category_id = ca.category_id
GROUP BY ca.name
ORDER BY total_sales DESC) #peguei do DER


SELECT sa.categoria,
       sa.total_sales,
       qtd.quant_disponivel,
       lj.vendas,
       lj.qual_loja,
       AVG(f.rental_rate) AS media_aluguel,
       AVG(f.replacement_cost) AS media_resposição
FROM sales sa
INNER JOIN qtd_disponivel qtd on sa.categoria=qtd.categoria
INNER JOIN loja lj on lj.categoria=qtd.categoria
INNER JOIN category c on c.name=lj.categoria
INNER JOIN film_category fc on fc.category_id=c.category_id
INNER JOIN film f on f.film_id=fc.film_id
GROUP by 1,5
ORDER BY 2 DESC;

/* 
QUESTÃO 2 de outra forma
*/

WITH loja_1 AS
(
SELECT 
     ca.name AS categoria, 
     count(r.rental_id) AS vendas_loja1
FROM category ca
 INNER JOIN film_category fc
    on fc.category_id=ca.category_id
     INNER JOIN film f
    on f.film_id=fc.film_id
     INNER JOIN inventory i 
    on i.film_id=f.film_id
     INNER JOIN rental r
    on r.inventory_id=i.inventory_id
    where store_id=1
GROUP BY 1)


,loja_2 AS
(
SELECT 
     ca.name AS categoria, 
     count(r.rental_id) AS vendas_loja2
FROM category ca
 INNER JOIN film_category fc
    on fc.category_id=ca.category_id
     INNER JOIN film f
    on f.film_id=fc.film_id
     INNER JOIN inventory i 
    on i.film_id=f.film_id
     INNER JOIN rental r
    on r.inventory_id=i.inventory_id
    where store_id=2
GROUP BY 1)

,qtd_disponivel AS
(
SELECT ca.name AS categoria,
       COUNT(i.inventory_id) AS 'quant_disponivel'
  FROM inventory i
   INNER JOIN film f
       on f.film_id=i.film_id
       INNER JOIN film_category fc
       ON f.film_id = fc.film_id
       INNER JOIN category ca
       ON ca.category_id = fc.category_id
       LEFT JOIN (SELECT inventory_id,
                         return_date
                    FROM rental
                   GROUP BY 1) comp
       ON i.inventory_id = comp.inventory_id
 WHERE comp.return_date IS NOT NULL # só estão disponivéis os que foram entregues
 GROUP BY 1
 ORDER BY 2 DESC
 )
 
 ,sales as
 (SELECT
 ca.name AS categoria,
    SUM(p.amount) AS total_sales
  FROM payment AS p
    INNER JOIN rental AS r ON p.rental_id = r.rental_id
    INNER JOIN inventory AS i ON r.inventory_id = i.inventory_id
INNER JOIN film AS f ON i.film_id = f.film_id
    INNER JOIN film_category AS fc ON f.film_id = fc.film_id
    INNER JOIN category AS ca ON fc.category_id = ca.category_id
GROUP BY ca.name
ORDER BY total_sales DESC) #peguei do DER


SELECT sa.categoria,
       sa.total_sales,
       qtd.quant_disponivel,
       lj1.vendas_loja1,
       lj2.vendas_loja2,
       AVG(f.rental_rate) AS media_aluguel,
       AVG(f.replacement_cost) AS media_reposição
FROM sales sa
INNER JOIN qtd_disponivel qtd on sa.categoria=qtd.categoria
INNER JOIN loja_1 lj1 on lj1.categoria=qtd.categoria
INNER JOIN loja_2 lj2 on lj2.categoria=lj1.categoria
INNER JOIN category c on c.name=lj2.categoria
INNER JOIN film_category fc on fc.category_id=c.category_id
INNER JOIN film f on f.film_id=fc.film_id
GROUP by 1
ORDER BY 2 DESC;

