(define (domain pancake)

    (:requirements :typing :arrays :bounded-integers)

    (:types
        pancakes - (number 0 9)
        stack    - (array 10 pancakes)
    )

    (:functions
        (pancake_stack) - stack
    )

    (:action flip
        :parameters (?f - pancakes)
        :precondition ()
        :effect (and
            (forall (?i - (number 0 ?f))
                (write
                    (pancake_stack) (?i)
                    (read (pancake_stack) (- ?f ?i))
                )
            )
        )
    )
)
