(define (problem sets-test-problem)
    (:domain sets-test)

    (:objects
        apple banana cherry orange - item
    )

    (:init
        (= (basket) (set.mk (cherry orange)))
    )

    (:goal
        (and
            ;(member apple (basket))
            ;(member banana (basket))
            ;(not (member cherry (basket)))
            (= (basket) (set.mk (apple banana)))
        )
    )
)
