/*Tarefa SQL Módulo 07

O conceito de relacionamentos entre tabelas por meio da sintaxe INNER JOIN abre
novas portas na sua jornada com SQL. Estude o conteúdo, e utilizando a base de
dados sakila responda às perguntas:

1 - 
Escreva uma consulta SQL que:
- Parta da tabela film (dica: essa será a tabela ao lado do comando
FROM)
- Relacione todas as tabelas com prefixo film_
- Não tenha colunas redundantes no resultado final (ou seja, as colunas em comum devem ser removidas da consulta)
*/
  

SELECT film.film_id, film.title, film.description, film.release_year, film.language_id, film.original_language_id, film.rental_duration, film.rental_rate, film.length, film.replacement_cost, film.rating, film.special_features, film.last_update, film.last_update
    FROM sakila.film 
    INNER JOIN sakila.film_actor 
    ON film_actor.film_id = film.film_id
    INNER JOIN sakila.film_category 
    ON film_category.film_id = film.film_id
    INNER JOIN sakila.film_text
    ON film_text.film_id = film.film_id;

/*
2. Esse mês, iremos oferecer um prêmio para todos os consumidores que já alugaram mais de 40 filmes! Para auxiliar na estratégia, a equipe responsável precisa da sua ajuda para identificar quem são essas pessoas, e gerar uma base com seus dados de contato (nome, sobrenome, email e cidade da loja de cadastro).
Dicas: sua consulta final deverá retornar as 4 colunas listadas acima, e
você precisará fazer a união entre diversas tabelas para isso. Caso
necessário, revise o módulo 5 - Funções de Agregação.
*/
SELECT 
c.first_name as'nome', c.last_name as 'sobrenome',c.email as 'email', ci.city as'cidade do endereço',
(CASE WHEN store_id=1 THEN 'Lethbridge'
ELSE 'Woodridge' END) as'cidade da loja de cadastro' FROM payment p
INNER JOIN customer c ON c.customer_id=p.customer_id 
INNER JOIN address ad on ad.address_id=c.address_id
INNER JOIN city ci on ci.city_id=ad.city_id

    GROUP BY c.customer_id HAVING COUNT(p.customer_id)>40

/*
3 – 
Crie uma consulta que relaciona a tabela inventory com outras tabelas. Com base no seu código e nos resultados, o que você imagina que essa tabela representa?
*/

SELECT inventory.inventory_id, inventory.film_id, inventory.store_id, rental.rental_id, rental.rental_date, rental.inventory_id, rental.customer_id, rental.return_date, rental.staff_id, rental.last_update
FROM inventory
    INNER JOIN rental ON inventory.inventory_id = rental.inventory_id


/*
4  - 
Com base na tabela rental, podemos ver que alguns aluguéis não foram
devolvidos (registros em que a coluna return_date é nula). Qual é o valor total pago por esses aluguéis?
*/
SELECT rental.return_date, SUM(payment.amount)
FROM sakila.rental
INNER JOIN sakila.payment
ON rental.rental_id=payment.rental_id
WHERE return_date IS NULL
