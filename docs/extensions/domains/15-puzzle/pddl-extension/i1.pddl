(define (problem i1)
    (:domain fifteen-puzzle)

    (:init
        (= (puzzle) (array.mk ((14 13 15  7)
                               (11 12  9  5)
                               ( 6  0  2  1)
                               ( 4  8 10  3)))
        )
    )

    (:goal
        (= (puzzle) (array.mk (( 0  1  2  3)
                               ( 4  5  6  7)
                               ( 8  9 10 11)
                               (12 13 14 15)))
        )
    )
)
