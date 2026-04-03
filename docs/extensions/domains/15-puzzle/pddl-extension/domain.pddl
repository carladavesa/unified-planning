(define (domain fifteen-puzzle)

    (:requirements :arrays)

    (:types
        size - (number 0 3)
        range - (number 0 15)
        puzzle15 - (array 4 4 range)
    )

    (:functions
        (puzzle) - puzzle15
    )

    ;; Tile at (?i, ?j) moves UP:   blank must be at (?i-1, ?j), i > 0
    (:action move_up
        :parameters (?i ?j - size)
        :precondition (and
            (= (read (puzzle) (- ?i 1) ?j) 0)
            (not (= (read (puzzle) ?i ?j) 0))
        )
        :effect (and
            (write ((puzzle) (- ?i 1) ?j) (read (puzzle) ?i ?j))
            (write ((puzzle) ?i ?j) 0)
        )
    )

    ;; Tile at (?i, ?j) moves DOWN:  blank must be at (?i+1, ?j), i < 3
    (:action move_down
        :parameters (?i ?j - size)
        :precondition (and
            (= (read (puzzle) (+ ?i 1) ?j) 0)
            (not (= (read (puzzle) ?i ?j) 0))
        )
        :effect (and
            (write ((puzzle) (+ ?i 1) ?j) (read (puzzle) ?i ?j))
            (write ((puzzle) ?i ?j) 0)
        )
    )

    ;; Tile at (?i, ?j) moves LEFT:  blank must be at (?i, ?j-1), j > 0
    (:action move_left
        :parameters (?i ?j - size)
        :precondition (and
            (= (read (puzzle) ?i (- ?j 1)) 0)
            (not (= (read (puzzle) ?i ?j) 0))
        )
        :effect (and
            (write ((puzzle) ?i (- ?j 1)) (read (puzzle) ?i ?j))
            (write ((puzzle) ?i ?j) 0)
        )
    )

    ;; Tile at (?i, ?j) moves RIGHT: blank must be at (?i, ?j+1), j < 3
    (:action move_right
        :parameters (?i ?j - size)
        :precondition (and
            (= (read (puzzle) ?i (+ ?j 1)) 0)
            (not (= (read (puzzle) ?i ?j) 0))
        )
        :effect (and
            (write ((puzzle) ?i (+ ?j 1)) (read (puzzle) ?i ?j))
            (write ((puzzle) ?i ?j) 0)
        )
    )
)
