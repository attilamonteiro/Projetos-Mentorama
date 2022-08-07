/*
-- Query: WITH sales_by_film_category AS 
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
-- Date: 2022-08-07 10:52
*/
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Action',4375.85,1.598696,21.179748,'loja 1',312);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Action',4375.85,1.598696,21.179748,'loja 2',312);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Animation',4656.30,7.561429,20.280738,'loja 1',335);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Animation',4656.30,7.561429,20.280738,'loja 2',335);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Children',3655.55,3.156667,20.057725,'loja 1',269);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Children',3655.55,3.156667,20.057725,'loja 2',269);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Classics',3639.59,3.990000,20.956986,'loja 1',270);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Classics',3639.59,3.990000,20.956986,'loja 2',270);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Comedy',4383.58,4.323333,19.018693,'loja 1',269);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Comedy',4383.58,4.323333,19.018693,'loja 2',269);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Documentary',4217.52,6.037619,20.725238,'loja 1',294);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Documentary',4217.52,6.037619,20.725238,'loja 2',294);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Drama',4587.39,5.523333,21.633396,'loja 1',300);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Drama',4587.39,5.523333,21.633396,'loja 2',300);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Family',4226.07,5.712222,20.006423,'loja 1',310);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Family',4226.07,5.712222,20.006423,'loja 2',310);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Foreign',4270.67,5.990000,18.615363,'loja 1',300);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Foreign',4270.67,5.990000,18.615363,'loja 2',300);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Games',4281.33,5.729130,20.736130,'loja 1',276);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Games',4281.33,5.729130,20.736130,'loja 2',276);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Horror',3722.54,1.490000,19.616478,'loja 1',248);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Horror',3722.54,1.490000,19.616478,'loja 2',248);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Music',3417.72,1.720769,19.188795,'loja 1',232);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Music',3417.72,1.720769,19.188795,'loja 2',232);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('New',4351.62,6.101111,19.727234,'loja 1',275);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('New',4351.62,6.101111,19.727234,'loja 2',275);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Sports',5314.21,4.126364,20.558278,'loja 1',344);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Sports',5314.21,4.126364,20.558278,'loja 2',344);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Travel',3549.64,3.775714,19.156069,'loja 1',235);
INSERT INTO `` (`category`,`total_sales`,`média_aluguel`,`média_reposição`,`loja`,`qtd_titulos`) VALUES ('Travel',3549.64,3.775714,19.156069,'loja 2',235);
