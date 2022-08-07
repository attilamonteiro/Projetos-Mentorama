/*
# A view chamada sales_by_film_category traz o resultado total de vends em alugéis por categoria dos filmes. Escreva uma consulta que complemente essa tabelacom as seguintes métricas para cada categoria:
#valor médio do aluguel dos filmes por categoria;
#valor médio do custo de reposição dos filmes por categoria;
#total de vendas por categoria na loja 1;
#total de vendas por categoria na loja 2;
#total de títulos disponíveis por categoria
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

