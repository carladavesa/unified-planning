(define (problem pancake_i0)
    (:domain pancake)

    (:init
        (= (pancake_stack) (array.mk (3 4 2 1 0)))
    )

    (:goal
        (= (pancake_stack) (array.mk (0 1 2 3 4)))
    )
)
