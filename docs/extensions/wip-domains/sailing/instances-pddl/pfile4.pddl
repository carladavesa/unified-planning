;; Enrico Scala (enricos83@gmail.com) and Miquel Ramirez (miquel.ramirez@gmail.com)
;;Setting seed to 1229
(define (problem instance151229)

	(:domain sailing)

	(:objects
		b0  - boat
		p0 p1 p2 p3 p4  - person
	)

  (:init
		(= (x b0) -14)
(= (y b0) 0)


		(= (d p0) -740)
(= (d p1) -116)
(= (d p2) 126)
(= (d p3) 966)
(= (d p4) 446)


	)

	(:goal
		(and
			(saved p0)
(saved p1)
(saved p2)
(saved p3)
(saved p4)

		)
	)
)


