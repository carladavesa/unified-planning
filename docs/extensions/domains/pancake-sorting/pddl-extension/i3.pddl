(define (problem pancake_i3)
    (:domain pancake)

    (:init
        (= (pancake_stack) (array.mk (1 4 0 3 2)))
    )

    (:goal
        (= (pancake_stack) (array.mk (0 1 2 3 4)))
    )
)
