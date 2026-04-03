(define (problem labyrinth-i1)
    (:domain labyrinth)

    (:init
        (= (robot_at) 0)

        (= ((card_at) 0 0) (set.mk (E W)))
        (= ((card_at) 1 0) (set.mk (S W)))
        (= ((card_at) 2 0) (set.mk (N W S E)))
        (= ((card_at) 3 0) (set.mk (N W E)))

        (= ((card_at) 0 1) (set.mk (N W E)))
        (= ((card_at) 1 1) (set.mk (S W)))
        (= ((card_at) 2 1) (set.mk (N S)))
        (= ((card_at) 3 1) (set.mk (S E)))

        (= ((card_at) 0 2) (set.mk (N S E)))
        (= ((card_at) 1 2) (set.mk (S W)))
        (= ((card_at) 2 2) (set.mk (N S E)))
        (= ((card_at) 3 2) (set.mk (W E)))

        (= ((card_at) 0 3) (set.mk (N W)))
        (= ((card_at) 1 3) (set.mk (N W S)))
        (= ((card_at) 2 3) (set.mk (N S)))
        (= ((card_at) 3 3) (set.mk (S W)))

        (= (card_at)
            (array.mk (
                        ((set.mk (E W)) (set.mk (S W)) (set.mk (N W S E)) (set.mk (N W E)))
                        ((set.mk (N W E)) (set.mk (S W)) (set.mk (N S)) (set.mk (S E)))
                        ((set.mk (N S E)) (set.mk (S W)) (set.mk (N S E)) (set.mk (W E)))
                        ((set.mk (N W)) (set.mk (N W S)) (set.mk (N S)) (set.mk (S W)))
                )
            )
        )

    )

    (:goal
        (and
            (= (robot_at) 15)
            (member S (read (card_at) 3 3))
        )
    )
)
