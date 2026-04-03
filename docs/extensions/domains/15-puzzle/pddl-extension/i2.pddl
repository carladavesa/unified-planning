(define (problem i2)
    (:domain fifteen-puzzle)

    (:init
        (= (puzzle) (array.mk ((13  5  4 10)
                               ( 9 12  8 14)
                               ( 2  3  7  1)
                               ( 0 15 11  6)))
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
