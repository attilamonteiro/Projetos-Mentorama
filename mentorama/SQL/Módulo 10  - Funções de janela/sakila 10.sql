/*Tarefa SQL Módulo 10
-- As funções de janela são uma aplicação avançada do uso do SQL para manipulação de dados. 
É uma grande ferramenta para possuir, que pode te ajudar a chegar mais rápido em análises estatísticas avançadas! 
Para solidificar seus conhecimentos, responda às perguntas abaixo. 
Submeta suas respostas em um arquivo com extensão .sql, e use comentários (com os caracteres -- ou /* *,/
para escrever suas respostas por extenso e delimitar as questões. Dica: utilize subconsultas para facilitar seu trabalho! 

1. Escreva uma consulta SQL que, com base no total em pagamentos realizados (coluna amount da tabela payment), 
divida todos os consumidores da locadora em 4 grupos. Além disso, com ajuda da função CASE WHEN, classifique consumidores nesses grupos, 
do de maiores pagadores ao de menores, respectivamente em: ○ “Especial” ○ “Fiel” ○ “Ocasional” ○ “Iniciante” Sua consulta final deve conter as colunas: email, classificacao, total_pago
-- 1-	  */

WITH consumidores AS (
SELECT c.email, 
       NTILE(4) OVER (ORDER BY amount ASC) AS grupo_preco,
       SUM(amount) OVER (ORDER BY amount) AS total_pago
  FROM payment
  INNER JOIN customer AS c USING (customer_id)
)

SELECT *,
       (CASE
         WHEN grupo_preco = 1 THEN "Iniciante"
         WHEN grupo_preco = 2 THEN "Ocasional"
         WHEN grupo_preco = 3 THEN "Fiel"
         WHEN grupo_preco = 4 THEN "Especial"
       END) AS nome_grupo_preco
  FROM consumidores;
/*
2-	
Escreva uma consulta SQL que responda: qual foi a primeira loja da rede a atingir um total de $10.000 no mês de julho/2005?
  */
  
SELECT s.store_id AS id_loja,
       p.amount as valor,
       p.payment_date as data_pagamento,
       SUM(p.amount) 
       OVER(PARTITION BY s.store_id
ORDER BY p.payment_date ROWS UNBOUNDED PRECEDING)
AS Valor_acumulativo
FROM store s
LEFT JOIN staff sa USING (store_id)
LEFT JOIN payment p USING (staff_id)
WHERE payment_date BETWEEN "2005-07-01" AND "2005-07-31"
ORDER BY Valor_acumulativo ASC

;
