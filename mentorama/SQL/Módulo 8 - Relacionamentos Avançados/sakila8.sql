/* Tarefa SQL Módulo 08
Além da sintaxe INNER JOIN, estudamos diversas outras opções de relacionamentos entre tabelas SQL, que a partir de agora te permitem explorar as mais diversas bases de dados. Não haverá um dia no seu trabalho com SQL em que você não irá fazer um relacionamento! Antes de caminharmos para o próximo módulo, estude o conteúdo e responda às perguntas: 
1. A gestão da locadora decidiu que no mês de julho de 2005 haverá um prêmio para o funcionário que mais registrou vendas, e pediu sua ajuda para determinar a premiação. 


a)	Usando a tabela staff como driving table, escreva uma consulta SQL que traga todas as vendas realizadas por cada funcionário no mês de julho de 2005. Não é permitido utilizar o filtro WHERE! 
*/
SELECT first_name, last_name, payment.amount, payment.payment_date
FROM sakila.staff
RIGHT JOIN sakila.payment
ON staff.staff_id=payment.staff_id
HAVING payment.payment_date BETWEEN '2005-07-01' AND '2005-07-31'
ORDER BY payment_date;
/*
b)  

Utilizei RIGHT JOIN (porém poderia ter usado left join ou inner join), pois caso utilizasse Union/Union ALL teríamos uma coluna embaixo da outra por exemplo, não representando os valores para cada funcionário.

c)  Altere essa consulta para trazer o total em pagamentos processados por cada funcionário
*/
SELECT first_name as Nome, SUM(payment.amount) as 'Total de venda'
FROM staff
LEFT JOIN payment
ON payment.staff_id = staff.staff_id
WHERE payment.payment_date BETWEEN "2005-07-01" AND "2005-07-31"
GROUP BY first_name 
ORDER BY payment.amount, payment.staff_id;

/*
d) Por fim, responda: qual funcionário deve ganhar o prêmio? Qual foi o valor total de vendas no mês?

Jon deve levar o premio pois, fez mais vendas com um total de 12852,86. Enquanto Mike fez 12652,82.
*/

/*
2 - Na tarefa do módulo anterior, você descobriu que alguns aluguéis ainda não foram devolvidos. Precisamos buscar esses itens! Escreva uma consulta que retorne a lista de e-mails das pessoas que estão com aluguéis pendentes, e o número de itens a serem retornados. Bônus opcional: utilize a função GROUP_CONCAT, que estudamos no módulo 5, para retornar também uma lista com o nome dos filmes que devem ser devolvidos.
*/
SELECT customer.email as EMAILS, GROUP_CONCAT(film.title) as Lista_filmes, COUNT(return_date is null) as Alugueis_a_devolver
FROM sakila.rental
LEFT JOIN sakila.payment USING (customer_id)
LEFT JOIN sakila.customer USING (customer_id)
LEFT JOIN sakila.inventory USING (inventory_id)
LEFT JOIN sakila.film USING (film_id)
WHERE return_date is null
GROUP By customer.email