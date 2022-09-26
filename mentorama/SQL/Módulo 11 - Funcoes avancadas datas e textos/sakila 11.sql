/*Nesse módulo, estudamos funções que ajudam a manipular dados de dois grandes tipos: datas e texto.
Nas tarefas abaixo, você terá a oportunidade de aplicar esse conhecimento junto com outros que já estudamos nos módulos anteriores, construindo consultas avançadas. Não deixe de utilizar as páginas de ajuda e documentação do MySQL caso precise consultar o formato de uso das funções!

1.	Conhecendo funções de datas, você pode agora conhecer mais sobre algumas métricas de performance de vendas das lojas. Escreva consultas que retornem:
a.	A quantidade de aluguéis por mês por loja.
*/

With qtd_por_mes as
(
SELECT s.store_id as loja, 
MONTH(r.rental_date) as mes, 
COUNT(r.rental_id) as quantidade
FROM rental r
LEFT JOIN staff sa on sa.staff_id=r.staff_id
INNER JOIN store s on sa.staff_id=s.manager_staff_id
GROUP BY 1, 2
ORDER BY 1)

SELECT loja, AVG(quantidade) as 'quantidade por loja'
from qtd_por_mes
GROUP BY loja

/*
b.	O valor total pago por mês em toda a rede.
*/

SELECT MONTH(p.payment_date) as 'mês',
SUM(p.amount) as 'total arrecadado'
FROM payment p
group by 1
ORDER BY 1 desc;

/*
c.	A quantidade de aluguéis por semana entre maio e junho de 2005. Qual semana teve a melhor performance?
*/
SELECT EXTRACT(WEEK FROM rental_date) as semana,
count(rental_id) as qtd_alugueis
FROM rental
WHERE MONTH(rental_date)<=6 and MONTH(rental_date)>=5
group by 1
order by 2 DESC;





/*
2.	A gerência da rede deseja entender se no mês de maio de 2005 houve diferença na recorrência dos consumidores fidelizados (ou seja, que realizaram mais de um aluguel) entre as duas lojas. Seu desafio é escrever uma consulta que retorne o número médio de dias entre os aluguéis desses consumidores, por loja que estão registrados. Dicas de como construir essa consulta:

-	Descubra, primeiro, quem são os consumidores (pelo seu customer_id) que queremos analisar: realizaram dois ou mais aluguéis no mês de maio de 2005.
-	Depois, avalie a diferença de tempo entre cada aluguel e o seguinte, por consumidor. Você precisará de uma função de janela para isso.
-	Por fim, calcule a média de tempo entre aluguéis para cada uma das lojas. Lembre-se que a tabela customer indica qual loja cada pessoa se cadastrou.

Com isso, responda: há diferença no número médio de dias entre aluguéis desse segmento de consumidores para as duas lojas?

RESPOSTA:
*/
-- Descubra, primeiro, quem são os consumidores (pelo seu customer_id) que queremos analisar: realizaram dois ou mais aluguéis no mês de maio de 2005.


WITH aluguel as 
(
SELECT customer_id , rental_date, COUNT(rental_id) OVER ( PARTITION BY customer_id) qtd_alugueis
from sakila.rental
WHERE rental_date BETWEEN "2005-05-01" AND "2005-05-31"
),

-- Depois, avalie a diferença de tempo entre cada aluguel e o seguinte, por consumidor. Você precisará de uma função de janela para isso.


datas AS 
(
SELECT t1.customer_id, 
rental_date AS aluguel, 
t2.store_id AS 'loja',
DATEDIFF(LEAD(rental_date) OVER (PARTITION BY t1.customer_id order by rental_date) , rental_date) AS 'diferenca'
FROM aluguel AS t1
INNER JOIN customer t2 ON t2.customer_id = t1.customer_id
WHERE qtd_alugueis >=2 
)

-- Por fim, calcule a média de tempo entre aluguéis para cada uma das lojas. Lembre-se que a tabela customer indica qual loja cada pessoa se cadastrou.

SELECT loja,
       AVG(diferenca)
  FROM datas
   GROUP BY loja;
;
/*
3.	Reescreva a consulta da tarefa do Módulo 3, dessa vez utilizando um filtro com expressões regulares: quais filmes disponíveis na locadora têm indicação de orientação parental (PG ou PG-13)?
*/
SELECT title, rating 
FROM sakila.film 
where rating = 'PG' or rating = 'PG-13' 
order by 1;
