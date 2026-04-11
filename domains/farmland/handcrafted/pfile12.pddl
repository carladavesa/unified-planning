;;Setting seed to 1229
;; Enrico Scala (enricos83@gmail.com) and Miquel Ramirez (miquel.ramirez@gmail.com)
(define (problem instance64001229ladder)
	(:domain farmland)
	(:objects
		farm0 farm1 farm2 farm3 farm4 farm5  - farm
	)
  (:init
		(= (x farm0) 400)
		(= (x farm1) 1)
		(= (x farm2) 0)
		(= (x farm3) 0)
		(= (x farm4) 0)
		(= (x farm5) 1)
		
		(adj farm0 farm1)
		(adj farm0 farm3)
		(adj farm1 farm0)
		(adj farm1 farm2)
		(adj farm1 farm4)
		(adj farm2 farm1)
		(adj farm2 farm5)
		(adj farm3 farm0)
		(adj farm3 farm4)
		(adj farm4 farm1)
		(adj farm4 farm3)
		(adj farm4 farm5)
		(adj farm5 farm2)
		(adj farm5 farm4)
		
		(= (cost) 0)
	)
	(:goal
		(and
			(>= (x farm0) 1)
			(>= (x farm1) 1)
			(>= (x farm2) 1)
			(>= (x farm3) 1)
			(>= (x farm4) 1)
			(>= (x farm5) 1)
			
			(>= (+ (* 10 (x farm0))(+ (* 17 (x farm1))(+ (* 13 (x farm2))(+ (* 11 (x farm3))(+ (* 14 (x farm4))(+ (* 19 (x farm5)) 0)))))) 5600)
		)
	)
)


