(define (problem pancake_i3)
    (:domain pancake)

    (:init
        (= (pancake_stack) (array.mk (0 2 3 4 1)))
    )

    (:goal
        (= (pancake_stack) (array.mk (0 1 2 3 4)))
    )
)
