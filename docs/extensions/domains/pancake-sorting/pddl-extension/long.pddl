(define (problem pancake_i3)
    (:domain pancake)

    (:init
        (= (pancake_stack) (array.mk (1 8 9 6)))
    )

    (:goal
        (= (pancake_stack) (array.mk (0 1 2 3 4 5 6 7 8 9)))
    )
)
