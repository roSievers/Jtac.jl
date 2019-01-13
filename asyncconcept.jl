# Test von unserem asynchronem Konzept

max_batch_size = 50

# enthält Tuple{Input, Channel{Output}}
channel = Channel(2 * max_batch_size)

model = (channel = channel, )

function calculate(model, x)
    sleep(1) # Das bildet unsere Situation noch nicht ganz ab. Eigentlich will ich drum herum locken/unlocken.
    x .* x
end

function async_calculate(model, x)
    out_channel = Channel(0)
    put!(model.channel, (x, out_channel))
    take!(out_channel)
end

second(tuple) = tuple[2]

# Runs in the background and processes batches.
function worker_thread(model; max_batch_size = 5)
    while true
        if closed_and_empty(model.channel)
            # No more work to be done, channel is empty
            println("Stopping.")
            return
        end
        inputs = Vector()
        # If we arrive here, there is at least one thing to be done.
        while isready(model.channel) && length(inputs) < max_batch_size
            push!(inputs, take!(model.channel))
        end
        # Here 1 <= length(inputs) <= max_batch_size
        println("Processing $(length(inputs)) inputs at the same time.")
        outputs = calculate(model, first.(inputs))
        for i = 1:length(inputs)
            # println("putting $(outputs[i]) into channel $(inputs[i][2]).")
            put!(inputs[i][2], outputs[i])
        end
    end
end

function closed_and_empty(channel)
    try fetch(channel); false catch _ true end
end


# Example
@async worker_thread(model, max_batch_size = max_batch_size)

sum(asyncmap( n -> async_calculate(model, n), 1:100 ))

close(model.channel)