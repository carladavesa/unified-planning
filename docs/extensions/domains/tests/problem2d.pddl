(define (problem test-2d-p1)
    (:domain test-2d)

    (:objects
        agent1 - agent
    )

    (:init
        (= (board agent1) (array.mk (0 0 0 0) (0 0 0 0) (0 0 0 0)))
    )

    (:goal
        (and
            (= (read (board agent1) (1) (2)) 2)
            (= (read (board agent1) (0) (3)) 1)
        )
    )

)
