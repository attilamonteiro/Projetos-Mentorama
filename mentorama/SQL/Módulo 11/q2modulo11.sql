/*
2.	A gerência da rede deseja entender se no mês de maio de 2005 houve diferença na recorrência dos consumidores fidelizados (ou seja, que realizaram mais de um aluguel) entre as duas lojas. Seu desafio é escrever uma consulta que retorne o número médio de dias entre os aluguéis desses consumidores, por loja que estão registrados. Dicas de como construir essa consulta:

-	Descubra, primeiro, quem são os consumidores (pelo seu customer_id) que queremos analisar: realizaram dois ou mais aluguéis no mês de maio de 2005.
-	Depois, avalie a diferença de tempo entre cada aluguel e o seguinte, por consumidor. Você precisará de uma função de janela para isso.
-	Por fim, calcule a média de tempo entre aluguéis para cada uma das lojas. Lembre-se que a tabela customer indica qual loja cada pessoa se cadastrou.

Com isso, responda: há diferença no número médio de dias entre aluguéis desse segmento de consumidores para as duas lojas?
*/


SELECT * , COUNT(rental_id) as alugueis , 
SELECT LEAD(aluguel) OVER ( PARTITION BY customer_id ORDER BY rental_date ROWS UNBOUNDED PRECEDING) as aluguel_anterior

from sakila.rental
WHERE rental_date BETWEEN "2005-05-01" AND "2005-05-31"
GROUP BY customer_id
HAVING COUNT(rental_id) >=2
order by COUNT(rental_id) ASC

datas AS (
SELECT aluguel,
       LEAD(aluguel) OVER (PARTITION BY customer_id ORDER BY alugueis DESC) AS aluguel_anterior
)

SELECT DATEDIFF(aluguel, aluguel_anterior)
  FROM datas