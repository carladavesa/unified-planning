(define (domain test-2d)

    (:requirements :typing :adl :fluents :numeric-fluents :arrays :bounded-integers)

    (:types
        agent   - object

        ;; Bounded integer dimensions
        row     - (number 0 2)
        col     - (number 0 3)

        ;; 2D array type: row x col grid
        grid    - (array 3 4 row)
    )

    (:functions
        ;; 2D array fluent: each agent has a 3x4 grid
        (board ?a - agent) - grid
    )

    ;; Write value ?v into cell (?i, ?j) of agent ?a's board.
    (:action fill
        :parameters (?a - agent ?i - row ?j - col ?v - row)
        :precondition (and
            (> ?v 0)
            (= (read (board ?a) ?i ?j) 0)
        )
        :effect (write ((board ?a) ?i ?j) ?v)
    )

    ;; Clear cell (?i, ?j): write 0 back.
    (:action clear
        :parameters (?a - agent ?i - row ?j - col)
        :precondition (> (read (board ?a) ?i ?j) 0)
        :effect (write ((board ?a) ?i ?j) 0)
    )

)
