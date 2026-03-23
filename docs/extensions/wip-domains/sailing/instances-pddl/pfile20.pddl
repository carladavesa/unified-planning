;; Enrico Scala (enricos83@gmail.com) and Miquel Ramirez (miquel.ramirez@gmail.com)
;;Setting seed to 1229
(define (problem instance4101229)

	(:domain sailing)

	(:objects
		b0 b1 b2 b3  - boat
		p0 p1 p2 p3 p4 p5 p6 p7 p8 p9  - person
	)

  (:init
		(= (x b0) -10)
(= (y b0) 0)
(= (x b1) 2)
(= (y b1) 0)
(= (x b2) -14)
(= (y b2) 0)
(= (x b3) 2)
(= (y b3) 0)


		(= (d p0) -740)
(= (d p1) -116)
(= (d p2) 126)
(= (d p3) 966)
(= (d p4) 446)
(= (d p5) 632)
(= (d p6) -788)
(= (d p7) -484)
(= (d p8) -320)
(= (d p9) -676)


	)

	(:goal
		(and
			(saved p0)
(saved p1)
(saved p2)
(saved p3)
(saved p4)
(saved p5)
(saved p6)
(saved p7)
(saved p8)
(saved p9)

		)
	)
)


